from typing import List
from langchain_core.prompts import ChatPromptTemplate

class QueryExpander:
    def __init__(self, llm):
        self.llm = llm

    def generate_queries(self, original_query: str, n_queries: int) -> List[str]:
        prompt = """
        你是一个查询扩展专家。给定用户的问题，生成 {n_queries} 个不同角度但语义相关的查询。
        
        原始问题: {query}
        
        要求：
        1. 保持语义相关性
        2. 从不同角度表达
        3. 每个查询要具体
        
        请直接返回查询列表，每行一个查询。
        """
        pt = ChatPromptTemplate.from_template(prompt)

        chain = pt | self.llm
        response = chain.invoke({"query": original_query, "n_queries": n_queries})
        queries = response.content.strip().split("\n")
        return [original_query] + queries

    def expand_query(self, original_query: str) -> List[str]:
        return self.generate_queries(original_query, 3)