import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate


class QueryExpander:
    def __init__(self, llm):
        self.llm = llm

    def _clean_queries(self, raw_lines: List[str], n_queries: int) -> List[str]:
        """清洗 LLM 输出的查询列表。"""
        blocked_terms = ("行业", "趋势", "技术方案", "可持续", "供应链", "创新方案")
        cleaned = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d]+[\.\)、]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            if any(term in line for term in blocked_terms):
                continue
            if line and len(line) <= 30:
                cleaned.append(line)
        return cleaned[:n_queries]

    def generate_queries(self, original_query: str, n_queries: int) -> List[str]:
        prompt = """你是电商客服系统的查询扩展助手。根据用户问题，生成 {n_queries} 个用户可能会问的客服问题。

原始问题: {query}

要求：
1. 场景限定在电商购物、售后、物流、订单、支付、会员等客服咨询
2. 每个查询不超过20字，用口语化的客服问题表达
3. 不要生成行业分析、技术趋势、宏观战略类内容
4. 若原始问题很短（如"物流"），扩展为具体客服问法，如"怎么查快递"、"运费多少"

请直接返回查询列表，每行一个查询，不要编号。"""
        pt = ChatPromptTemplate.from_template(prompt)

        chain = pt | self.llm
        response = chain.invoke({"query": original_query, "n_queries": n_queries})
        content = response.content if hasattr(response, "content") else str(response)
        raw_lines = content.strip().split("\n")
        expanded = self._clean_queries(raw_lines, n_queries)

        queries = [original_query]
        for q in expanded:
            if q != original_query and q not in queries:
                queries.append(q)
        return queries

    def expand_query(self, original_query: str) -> List[str]:
        return self.generate_queries(original_query, 3)
