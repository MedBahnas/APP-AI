from src.chatbot import Chatbot


class DummyVectorStore:
    def search(self, query, k=3):
        return [{'text': 'info', 'metadata': {'title': 'Doc1'}, 'start': 0, 'end': 4}]


def test_process_query(monkeypatch):
    vs = DummyVectorStore()
    bot = Chatbot(vs, model='test')

    def fake_generate(self, query, context=""):
        return f"answer:{query}"

    monkeypatch.setattr(Chatbot, '_generate_response', fake_generate)

    result = bot.process_query('hello')
    assert result['response'] == 'answer:hello'
    assert result['sources'] == ['Doc1']

