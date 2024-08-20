import torch
from transformers import AutoModel, AutoTokenizer


class E5Model:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_queries(self, queries: list[str]) -> torch.Tensor:
        queries = [f"query: {query}" for query in queries]
        batch_dict = self.tokenizer(
            queries, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**batch_dict)
        query_embedding = self._average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        return query_embedding

    def embed_docs(self, docs: list[str]) -> torch.Tensor:
        docs = [f"passage: {doc}" for doc in docs]
        batch_dict = self.tokenizer(
            docs, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**batch_dict)
        doc_embeddings = self._average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        return doc_embeddings

    def get_top_k_docs(
        self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor, k: int
    ) -> list[int]:
        similarities = torch.mm(doc_embeddings, query_embedding.T)
        top_similarities, top_indices = torch.topk(similarities.squeeze(), k=k)
        return [top_indices[i].item() for i in range(k)]

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
