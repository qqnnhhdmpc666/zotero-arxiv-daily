from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
import json
import os
from datetime import datetime
from pathlib import Path
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]

    def build_interest_seed_corpus(self) -> list[CorpusPaper]:
        seeds = self.config.zotero.get("interest_seed", [])
        if not seeds:
            return []
        now = datetime.utcnow()
        corpus = [
            CorpusPaper(
                title=str(seed.get("title", "Research interest seed")),
                abstract=str(seed.get("abstract", "")),
                added_date=now,
                paths=["interest_seed"],
            )
            for seed in seeds
            if seed.get("abstract")
        ]
        logger.info(f"Using {len(corpus)} configured interest seed papers as fallback corpus")
        return corpus
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def _parse_bool(self, value, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _history_path(self) -> Path:
        return Path(self.config.executor.get("recommendation_history_path", ".github/recommendation_history.json"))

    def _paper_history_key(self, paper: Paper) -> str:
        return (paper.url or paper.pdf_url or paper.title).strip().lower()

    def load_recommendation_history(self) -> set[str]:
        history_path = self._history_path()
        if not history_path.exists():
            return set()
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read recommendation history {history_path}: {e}")
            return set()

        records = history.get("recommended", history if isinstance(history, list) else [])
        keys = set()
        for record in records:
            if isinstance(record, str):
                keys.add(record.strip().lower())
            elif isinstance(record, dict) and record.get("key"):
                keys.add(str(record["key"]).strip().lower())
        logger.info(f"Loaded {len(keys)} previously recommended papers from {history_path}")
        return keys

    def save_recommendation_history(self, papers: list[Paper]) -> None:
        history_path = self._history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        existing_records = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                existing_records = history.get("recommended", history if isinstance(history, list) else [])
            except Exception as e:
                logger.warning(f"Failed to load existing recommendation history before saving: {e}")

        seen = set()
        records = []
        for record in existing_records:
            key = None
            if isinstance(record, str):
                key = record.strip().lower()
                record = {"key": key}
            elif isinstance(record, dict) and record.get("key"):
                key = str(record["key"]).strip().lower()
            if key and key not in seen:
                seen.add(key)
                records.append(record)

        today = datetime.utcnow().strftime("%Y-%m-%d")
        for paper in papers:
            key = self._paper_history_key(paper)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "key": key,
                "title": paper.title,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "code_url": paper.code_url,
                "source": paper.source,
                "score": paper.score,
                "abstract": paper.abstract,
                "tldr": paper.tldr,
                "recommended_at": today,
            })

        history_path.write_text(
            json.dumps({"recommended": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Saved {len(records)} recommendation history records to {history_path}")

    def select_recommendations(self, papers: list[Paper]) -> list[Paper]:
        max_paper_num = int(self.config.executor.max_paper_num)
        if max_paper_num <= 0 or not papers:
            return []

        include_top = self._parse_bool(
            self.config.executor.get("always_include_top_similarity", os.getenv("INCLUDE_TOP_SIMILARITY", True)),
            default=True,
        )
        history = self.load_recommendation_history()
        selected = []
        candidates = papers

        if include_top:
            selected.append(papers[0])
            candidates = papers[1:]
            logger.info("Always including the top-similarity paper, even if it was recommended before.")

        for paper in candidates:
            if len(selected) >= max_paper_num:
                break
            key = self._paper_history_key(paper)
            if key in history:
                logger.info(f"Skip previously recommended paper: {paper.title}")
                continue
            selected.append(paper)

        if len(selected) < min(max_paper_num, len(papers)):
            logger.warning(
                f"Selected {len(selected)} papers after de-duplication; "
                "not filling the remaining slots with previously recommended papers."
            )
        return selected

    
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.warning("No Zotero papers found after filtering; falling back to configured interest seeds.")
            corpus = self.build_interest_seed_corpus()
        if len(corpus) == 0:
            logger.error(f"No zotero papers or interest seeds found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = self.select_recommendations(reranked_papers)
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
        if len(reranked_papers) > 0:
            self.save_recommendation_history(reranked_papers)
