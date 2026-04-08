import os
import json
import argparse
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 清除可能的代理设置（避免代理影响 API 访问）
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]
        print(f"[环境] 已清除环境变量：{proxy_var}")

# 加载环境变量（优先使用项目目录下 .env，并覆盖系统同名变量）
load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=True)

from src.preprocessor import DataPreprocessor
from src.llm_extractor import DeepSeekExtractor
from src.lightrag_engine import create_lightrag_engine
from src.schemas import TripletExtractionResult
from pydantic import ValidationError


def _validate_api_key() -> str:
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise EnvironmentError("未检测到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件或当前环境变量配置")
    lowered = api_key.lower()
    placeholders = [
        "your_api_key_here",
        "your_deepseek_api_key_here",
        "dummy",
        "test",
        "example"
    ]
    if any(token in lowered for token in placeholders):
        raise EnvironmentError("检测到 DEEPSEEK_API_KEY 为占位符，请替换为真实 DeepSeek API Key")
    return api_key


def main(
    input_path: str = "data/input",
    top_event: Optional[str] = None,
    semantic_merge_threshold: float = 0.86,
    enable_lightrag: bool = False,
    lightrag_working_dir: str = "data/lightrag",
    lightrag_query: Optional[str] = None,
    lightrag_mode: str = "hybrid",
    enable_lightrag_refine: bool = False,
    refine_hops: int = 2,
    refine_parent_topk: int = 30,
    refine_merge_policy: str = "balanced",
):
    start_time = time.time()
    print("=" * 60)
    print("🚀 三元组抽取任务开始")
    print("=" * 60)
    
    _validate_api_key()
    normalized_top_event = (top_event or "").strip() or None

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/4] 初始化预处理器和抽取器...")
    init_start = time.time()
    preprocessor = DataPreprocessor(use_multimodal_encoding=False)
    extractor = DeepSeekExtractor(semantic_merge_threshold=semantic_merge_threshold)
    rag_engine = create_lightrag_engine(lightrag_working_dir) if enable_lightrag else None
    print(f"✓ 初始化完成 (耗时：{time.time() - init_start:.2f}s)")
    if normalized_top_event:
        print(f"🎯 合并目标顶事件：{normalized_top_event}")
    else:
        print("ℹ 未指定顶事件，将执行全量融合")
    print(f"🔧 语义合并阈值：{semantic_merge_threshold:.2f}")
    if enable_lightrag:
        print(f"🕸 LightRAG 引擎已启用：{lightrag_working_dir}")
        print(f"🔍 LightRAG 查询模式：{lightrag_mode}")
        if lightrag_query:
            print(f"❓ LightRAG 查询语句：{lightrag_query}")
        if enable_lightrag_refine:
            print(
                f"🎯 LightRAG 引导精抽已启用：hops={max(1, int(refine_hops))} "
                f"parent_topk={max(1, int(refine_parent_topk))} policy={refine_merge_policy}"
            )

    if os.path.isfile(input_path):
        files_to_process = [input_path]
        print(f"📁 检测到单个文件：{input_path}")
    elif os.path.isdir(input_path):
        files_to_process = [
            os.path.join(input_path, filename)
            for filename in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, filename))
        ]
        print(f"📂 检测到目录，共 {len(files_to_process)} 个文件")
    else:
        raise FileNotFoundError(f"输入路径不存在或不可用：{input_path}")

    for idx, file_path in enumerate(files_to_process, 1):
        filename = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"正在处理文件 [{idx}/{len(files_to_process)}]: {filename}")
        print(f"{'='*60}")
        
        file_start = time.time()
        
        try:
            # 1. 预处理提取文本
            print(f"\n[步骤 1/4] 预处理：提取文本内容...")
            preprocess_start = time.time()
            text = preprocessor.process_file(file_path)
            print(f"✓ 文本提取完成 (耗时：{time.time() - preprocess_start:.2f}s)")
            
            # 检查预处理结果
            if isinstance(text, dict):
                if not text.get('success'):
                    print(f"⚠ 预处理失败：{text.get('error')}")
                    continue
                text_content = text.get('text_content', '')
            else:
                text_content = text
            
            print(f"📊 提取到的文本长度：{len(text_content)} 字符")
            
            if not text_content.strip():
                print("⚠ 警告：提取到的文本内容为空！")
                continue

            # 2. 调用模型进行抽取
            print(f"\n[步骤 2/4] LLM 抽取：调用 DEEPSEEK API 进行三元组抽取...")
            extract_start = time.time()
            debug_log_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_triplets_debug.jsonl")
            json_result = extractor.extract(
                text_content,
                source_reference=filename,
                top_event=normalized_top_event,
                debug_log_path=debug_log_file
            )
            print(f"✓ LLM 抽取完成 (耗时：{time.time() - extract_start:.2f}s)")
            print(f"🧾 调试日志路径：{debug_log_file}")

            # 3. 校验并解析 JSON
            print(f"\n[步骤 3/4] 验证：解析并校验 JSON 格式...")
            validate_start = time.time()
            parsed_data = json.loads(json_result)
            validated_data = TripletExtractionResult(**parsed_data)
            triplet_count = len(validated_data.triplets)
            print(f"✓ JSON 验证成功，共抽取 {triplet_count} 个三元组 (耗时：{time.time() - validate_start:.2f}s)")
            output_payload = validated_data.model_dump(mode="json", exclude_none=True)

            lightrag_query_result = None
            if enable_lightrag and rag_engine is not None:
                ingest_start = time.time()
                ingest_stat = rag_engine.ingest_graph(
                    output_payload.get("lightrag_graph") or {},
                    source_reference=filename
                )
                print(
                    f"✓ LightRAG 入库完成 (耗时：{time.time() - ingest_start:.2f}s) "
                    f"文档片段数：{ingest_stat.get('inserted_documents', 0)}"
                )
                if lightrag_query:
                    query_start = time.time()
                    lightrag_query_result = rag_engine.query(
                        lightrag_query,
                        mode=lightrag_mode
                    )
                    print(f"✓ LightRAG 查询完成 (耗时：{time.time() - query_start:.2f}s)")
                if enable_lightrag_refine:
                    refine_start = time.time()
                    refine_seed = (normalized_top_event or lightrag_query or "").strip()
                    if refine_seed:
                        subgraph = rag_engine.retrieve_subgraph(
                            top_event=refine_seed,
                            hops=max(1, int(refine_hops)),
                            limit=max(1, int(refine_parent_topk))
                        )
                        candidate_parent_ids = rag_engine.collect_candidate_parent_ids(subgraph)
                        candidate_parent_ids = candidate_parent_ids[: max(1, int(refine_parent_topk))]
                        refined_triplets = extractor.refine_with_candidates(
                            text=text_content,
                            source_reference=filename,
                            index_data=output_payload.get("parent_child_index") or {},
                            candidate_parent_ids=candidate_parent_ids,
                            top_event=normalized_top_event
                        )
                        merged_triplets, merge_stats = extractor.merge_with_refinement(
                            base_triplets=output_payload.get("triplets") or [],
                            refined_triplets=refined_triplets,
                            index_data=output_payload.get("parent_child_index") or {},
                            candidate_parent_ids=candidate_parent_ids,
                            policy=refine_merge_policy
                        )
                        output_payload["triplets"] = merged_triplets
                        output_payload["lightrag_graph"] = extractor._build_lightrag_graph(merged_triplets, filename)
                        output_payload["retrieval_trace"] = subgraph
                        output_payload["refine_stats"] = {
                            **merge_stats,
                            "enabled": True,
                            "seed": refine_seed,
                            "refine_hops": max(1, int(refine_hops)),
                            "refine_parent_topk": max(1, int(refine_parent_topk)),
                            "candidate_parent_ids": candidate_parent_ids,
                            "latency_seconds": round(time.time() - refine_start, 4),
                        }
                        print(
                            f"✓ LightRAG 引导精抽完成 (耗时：{time.time() - refine_start:.2f}s) "
                            f"候选父块={len(candidate_parent_ids)} 融合后三元组={len(merged_triplets)}"
                        )
                    else:
                        output_payload["retrieval_trace"] = {
                            "nodes": [],
                            "edges": [],
                            "parent_ids": [],
                            "retrieval_trace": [],
                            "reason": "missing_refine_seed"
                        }
                        output_payload["refine_stats"] = {
                            "enabled": True,
                            "skipped": True,
                            "reason": "top_event_and_lightrag_query_both_empty",
                            "policy": (refine_merge_policy or "balanced").strip().lower() or "balanced"
                        }
                        print("⚠ LightRAG 引导精抽已启用，但未提供 top_event 或 lightrag_query，已跳过")

            print(f"\n[步骤 4/4] 保存：写入输出文件...")
            save_start = time.time()
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_triplets.json")
            if lightrag_query_result is not None:
                output_payload["lightrag_query"] = {
                    "question": lightrag_query,
                    "mode": lightrag_mode,
                    "result": lightrag_query_result
                }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_payload, f, ensure_ascii=False, indent=4)
            print(f"✓ 文件保存成功 (耗时：{time.time() - save_start:.2f}s)")
            print(f"📄 输出路径：{output_file}")

            print(f"\n✅ 文件处理完成！总耗时：{time.time() - file_start:.2f}s")

        except ValidationError as e:
            print(f"\n❌ Schema 验证失败：{e}\n")
        except Exception as e:
            print(f"\n❌ 处理失败，错误类型：{type(e).__name__}")
            print(f"错误详情：{e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"🎉 所有任务完成！总耗时：{time.time() - start_time:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从指定文件或目录中提取三元组信息")
    parser.add_argument(
        "-i",
        "--input-path",
        default="data/input",
        help="待处理输入文件路径，可传单个文件路径或目录路径",
    )
    parser.add_argument(
        "-t",
        "--top-event",
        default=None,
        help="可选：指定顶事件名称，仅围绕该顶事件进行全局合并与链路收敛",
    )
    parser.add_argument(
        "--semantic-merge-threshold",
        type=float,
        default=0.86,
        help="语义去重阈值（0~1），阈值越低越容易将书面差异合并为同一事件",
    )
    parser.add_argument(
        "--enable-lightrag",
        action="store_true",
        help="启用 LightRAG 引擎，自动将 lightrag_graph 入库",
    )
    parser.add_argument(
        "--lightrag-working-dir",
        default="data/lightrag",
        help="LightRAG 本地工作目录",
    )
    parser.add_argument(
        "--lightrag-query",
        default=None,
        help="可选：入库后立即执行一次 LightRAG 查询",
    )
    parser.add_argument(
        "--lightrag-mode",
        default="hybrid",
        help="LightRAG 查询模式，如 naive/local/global/hybrid",
    )
    parser.add_argument(
        "--enable-lightrag-refine",
        action="store_true",
        help="启用 LightRAG 图检索引导的二轮精抽流程",
    )
    parser.add_argument(
        "--refine-hops",
        type=int,
        default=2,
        help="图检索子图扩展跳数（多跳回溯）",
    )
    parser.add_argument(
        "--refine-parent-topk",
        type=int,
        default=30,
        help="参与二轮精抽的候选父块上限",
    )
    parser.add_argument(
        "--refine-merge-policy",
        choices=["strict", "balanced"],
        default="balanced",
        help="一轮与二轮融合策略：strict 或 balanced",
    )
    args = parser.parse_args()
    threshold = max(0.0, min(1.0, float(args.semantic_merge_threshold)))
    main(
        input_path=args.input_path,
        top_event=args.top_event,
        semantic_merge_threshold=threshold,
        enable_lightrag=bool(args.enable_lightrag),
        lightrag_working_dir=args.lightrag_working_dir,
        lightrag_query=args.lightrag_query,
        lightrag_mode=args.lightrag_mode,
        enable_lightrag_refine=bool(args.enable_lightrag_refine),
        refine_hops=max(1, int(args.refine_hops)),
        refine_parent_topk=max(1, int(args.refine_parent_topk)),
        refine_merge_policy=args.refine_merge_policy,
    )
