import json
from time import perf_counter
from telecom_support.taxonomy import TAXONOMY
from telecom_support.ingestion.chunking import token_count
from telecom_support.classification.service import classify
from telecom_support.classification.schemas import ComplaintLabels
from telecom_support.retrieval.service import retrieve
from telecom_support.generation.service import generate
from telecom_support.llm import ServiceError

class ResolutionPipeline:
    def __init__(self, root, index, llm, threshold=0.30):
        self.index, self.llm, self.threshold = index, llm, threshold
        data = json.loads((root/'data/indexes/classification_examples.json').read_text(encoding='utf-8'))
        if not data['examples']:
            raise ValueError('Regenerate classification examples for the current taxonomy.')
        # Additive labels preserve valid existing examples; new classes use definitions.
        self.taxonomy_version = TAXONOMY['version']
        self.examples = data['examples']
        for example in self.examples:
            ComplaintLabels.model_validate(example['labels'])

    def resolve(self, complaint):
        complaint = complaint.strip()
        if not complaint:
            raise ServiceError('Enter a customer complaint.')
        if token_count(self.index.embedder.tokenizer, complaint) > self.index.embedder.model.max_seq_length:
            raise ServiceError('Complaint exceeds the embedding limit. Enter a shorter complaint for this prototype.')
        start = perf_counter()
        labels = classify(self.llm,complaint,self.examples)
        classified = perf_counter()
        evidence, diagnostics = retrieve(self.index,complaint,labels,self.threshold)
        retrieved = perf_counter()
        resolution = generate(self.llm,complaint,labels,evidence)
        return {'classification':labels.model_dump(mode='json'),
                'resolution':resolution.model_dump(mode='json'), 'sources':evidence,
                'retrieval':diagnostics, 'timings_seconds':{
                    'classification':round(classified-start,3),
                    'retrieval':round(retrieved-classified,3),
                    'generation':round(perf_counter()-retrieved,3),
                    'total':round(perf_counter()-start,3)},
                'notice':'Agent-reviewed draft. Citation IDs are checked; claim support is not automatically proven.'}
