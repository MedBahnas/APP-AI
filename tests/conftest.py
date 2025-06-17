import sys
import types
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
# --- Minimal numpy stub ---
class SimpleArray(list):
    @property
    def shape(self):
        if not self:
            return (0,)
        first = self[0]
        if isinstance(first, (list, tuple, SimpleArray)):
            return (len(self), len(first))
        return (len(self),)

    def astype(self, dtype):
        return self

def array(data, dtype=None):
    if isinstance(data, SimpleArray):
        return data
    if not isinstance(data, (list, tuple)):
        data = [data]
    processed = []
    for item in data:
        if isinstance(item, (list, tuple, SimpleArray)):
            processed.append(list(item))
        else:
            processed.append(item)
    return SimpleArray(processed)

def vstack(arrays):
    result = []
    for arr in arrays:
        result.extend(list(arr))
    return SimpleArray(result)

def expand_dims(arr, axis):
    if axis == 0:
        return SimpleArray([arr])
    raise NotImplementedError

numpy_stub = types.ModuleType('numpy')
numpy_stub.array = array
numpy_stub.float32 = 'float32'
numpy_stub.vstack = vstack
numpy_stub.expand_dims = expand_dims
numpy_stub.ndarray = SimpleArray
sys.modules.setdefault('numpy', numpy_stub)

# --- Fake faiss module ---
class FakeIndexFlatL2:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vecs):
        for v in vecs:
            self.vectors.append(list(v))

    def search(self, queries, k):
        q = list(queries[0])
        distances = []
        for v in self.vectors:
            dist = sum((q[i] - v[i]) ** 2 for i in range(self.dim))
            distances.append(dist)
        idxs = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
        return [[distances[i] for i in idxs]], [idxs]

faiss_stub = types.ModuleType('faiss')
faiss_stub.IndexFlatL2 = FakeIndexFlatL2
sys.modules.setdefault('faiss', faiss_stub)

# --- Other external stubs ---
class FakeSentenceTransformer:
    def __init__(self, name):
        pass
    def encode(self, text, convert_to_numpy=True):
        return [1.0]

stub_st = types.ModuleType('sentence_transformers')
stub_st.SentenceTransformer = FakeSentenceTransformer
sys.modules.setdefault('sentence_transformers', stub_st)

class FakeOpenAI:
    def __init__(self, api_key=None):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kwargs: types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="stub"))])))

openai_stub = types.ModuleType('openai')
openai_stub.OpenAI = FakeOpenAI
openai_stub.api_key = None
openai_stub.Embedding = types.SimpleNamespace(create=lambda **kwargs: {'data':[{'embedding':[1.0]}]})
sys.modules.setdefault('openai', openai_stub)

for mod_name in ['PyPDF2', 'docx', 'pptx', 'magic']:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if mod_name == 'magic':
            class Magic:
                def __init__(self, mime=True):
                    pass
                def from_file(self, fp):
                    return 'text/plain'
            m.Magic = Magic
        elif mod_name == 'docx':
            class Document:
                def __init__(self, path):
                    self.paragraphs = []
            m.Document = Document
        elif mod_name == 'pptx':
            class Presentation:
                def __init__(self, path):
                    self.slides = []
            m.Presentation = Presentation
        elif mod_name == 'PyPDF2':
            class PdfReader:
                def __init__(self, file):
                    self.pages = []
            m.PdfReader = PdfReader
        sys.modules[mod_name] = m

if 'dotenv' not in sys.modules:
    dotenv_stub = types.ModuleType('dotenv')
    dotenv_stub.load_dotenv = lambda: None
    sys.modules['dotenv'] = dotenv_stub

os.environ.setdefault('OPENAI_API_KEY', 'test-key')
