import json
import os
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError

class StandardQAPair(BaseModel):
    """
    下游评测管线所需的标准 QA 结构体
    利用 Pydantic 实行字段类型检查与强制清洗
    """
    id: str = Field(..., description="唯一问答对 ID")
    query: str = Field(..., min_length=2, description="用户问题/Prompt")
    response: str = Field(..., min_length=2, description="模型回答或基准参考回答")
    source: str = Field(default="unknown", description="数据来源")

    @classmethod
    def clean_text(cls, text: Any) -> str:
        """剥离无效空白与非法转义字符"""
        if not isinstance(text, str):
            return ""
        return text.strip()


class DataCleanPipeline:
    def __init__(self, raw_path: str, output_path: str):
        self.raw_path = raw_path
        self.output_path = output_path
        self.stats: Dict[str, int] = {
            "total_lines": 0,
            "parsed_json_errors": 0,
            "schema_validation_errors": 0,
            "empty_field_filtered": 0,
            "valid_passed": 0
        }

    def process_line(self, line: str, line_no: int) -> Tuple[bool, StandardQAPair | None, str]:
        """流式处理单行数据，隔离解析崩溃与校验失败"""
        clean_line = line.strip()
        if not clean_line:
            return False, None, "空行跳过"

        # 1. 尝试 JSON 格式解析
        try:
            record: Dict[str, Any] = json.loads(clean_line)
        except json.JSONDecodeError as e:
            self.stats["parsed_json_errors"] += 1
            return False, None, f"JSON 解码失败: {str(e)}"

        # 2. 基础文本清洗：去除首尾空格、检测类型
        raw_query = StandardQAPair.clean_text(record.get("query"))
        raw_resp = StandardQAPair.clean_text(record.get("response"))

        # 3. 核心业务规则：过滤无效短文本
        if not raw_query or not raw_resp:
            self.stats["empty_field_filtered"] += 1
            return False, None, "query 或 response 为空/纯空格"

        # 4. Pydantic 强校验并注入规范化结构
        try:
            qa_pair = StandardQAPair(
                id=str(record.get("id", f"auto_gen_{line_no}")),
                query=raw_query,
                response=raw_resp,
                source=str(record.get("source", "unknown"))
            )
            return True, qa_pair, "校验通过"
        except ValidationError as ve:
            self.stats["schema_validation_errors"] += 1
            return False, None, f"Pydantic Schema 校验失败: {ve.error_count()} 项错误"

    def run(self):
        """执行端到端数据清洗流程"""
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"原始数据文件不存在: {self.raw_path}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        print(f"[*] 开始清洗任务: {self.raw_path}")
        with open(self.raw_path, "r", encoding="utf-8") as fin, \
             open(self.output_path, "w", encoding="utf-8") as fout:

            for line_no, line in enumerate(fin, start=1):
                self.stats["total_lines"] += 1
                is_valid, qa_obj, log_msg = self.process_line(line, line_no)

                if is_valid and qa_obj:
                    # 规范序列化写出为 JSONL
                    fout.write(qa_obj.model_dump_json() + "\n")
                    self.stats["valid_passed"] += 1
                else:
                    print(f"    [丢弃] 第 {line_no} 行: {log_msg}")

        self._print_summary()

    def _print_summary(self):
        """打印工程化清洗报告"""
        print("\n" + "="*40)
        print("          清洗流程审计报告")
        print("="*40)
        print(f"读取总行数 (Total Lines):          {self.stats['total_lines']}")
        print(f"JSON 损坏丢弃 (JSON Errors):        {self.stats['parsed_json_errors']}")
        print(f"空文本/过短丢弃 (Empty Filtered):    {self.stats['empty_field_filtered']}")
        print(f"模式校验不合规 (Schema Errors):     {self.stats['schema_validation_errors']}")
        print(f"最终有效入库 (Valid Output):        {self.stats['valid_passed']}")

        pass_rate = (self.stats["valid_passed"] / self.stats["total_lines"]) * 100 if self.stats["total_lines"] > 0 else 0
        print(f"数据清洗合格率 (Pass Rate):         {pass_rate:.2f}%")
        print("="*40)
        print(f"[+] 清洗后文件已保存至: {self.output_path}\n")


if __name__ == "__main__":
    RAW_FILE = os.path.join("data", "raw", "sample_qa_dirty.jsonl")
    OUTPUT_FILE = os.path.join("data", "processed", "clean_qa.jsonl")

    pipeline = DataCleanPipeline(raw_path=RAW_FILE, output_path=OUTPUT_FILE)
    pipeline.run()
