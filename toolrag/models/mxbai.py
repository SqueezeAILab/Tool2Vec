import torch
from transformers import AutoModel, AutoTokenizer


class MxbaiModel:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    # For retrieval you need to pass this prompt. Please find our more in our blog post.
    def _transform_query(self, query: str) -> str:
        """For retrieval, add the prompt for query (not for documents)."""
        return f"Represent this sentence for searching relevant passages: {query}"

    # The model works really well with cls pooling (default) but also with mean pooling.
    def _pooling(
        self, outputs: torch.Tensor, inputs: dict, strategy: str = "cls"
    ) -> torch.Tensor:
        if strategy == "cls":
            outputs = outputs[:, 0]
        elif strategy == "mean":
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1
            ) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs

    def embed_queries(self, queries: list[str]) -> torch.Tensor:
        queries = [self._transform_query(query) for query in queries]
        batch_dict = self.tokenizer(
            queries, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**batch_dict).last_hidden_state
        query_embedding = self._pooling(outputs, batch_dict, "cls")
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        return query_embedding

    def embed_docs(self, docs: list[str]) -> torch.Tensor:
        print(len(docs[0]))
        batch_dict = self.tokenizer(
            docs, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**batch_dict).last_hidden_state
        doc_embeddings = self._pooling(outputs, batch_dict, "cls")
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
