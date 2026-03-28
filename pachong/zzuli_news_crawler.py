# -*- coding: utf-8 -*-
"""
郑州轻工业大学官网新闻爬虫（支持断点续爬 + 优雅停止）

功能概览：
1. 起始 URL: https://www.zzuli.edu.cn/
2. 仅爬取同域名下以 .html/.htm 结尾的页面（BFS）
3. 严格从 <div class="article"> 提取指定字段
4. 存入 MySQL test.news（自动建库建表，url 唯一）
5. 控制台输入 1 回车后，完成当前页面后优雅退出
6. 通过本地状态文件保存待爬队列，实现断点续爬

python D:\code\CampusKnowledge_QASystem\pachong\zzuli_news_crawler.py

依赖：
    pip install requests beautifulsoup4 pymysql
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Set
from urllib.parse import urljoin, urlparse

import pymysql
import requests
from bs4 import BeautifulSoup
from pymysql.err import MySQLError


# ========================= 配置区域（可按需修改） =========================
START_URL = "https://www.zzuli.edu.cn/"
ALLOWED_DOMAIN = "zzuli.edu.cn"
MAX_WORKERS = 32
REQUEST_INTERVAL_SECONDS = 0.0
DB_BATCH_SIZE = 128
STATE_SAVE_EVERY_BATCHES = 5
STATE_FILE = Path(__file__).resolve().parent / "crawler_state.json"

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "123456"
DB_NAME = "campus_qa_db"
DB_TABLE = "campus_structured_data"

REQUEST_TIMEOUT = 15


@dataclass
class ArticleData:
    """承载单篇文章的结构化字段。"""

    title: str
    contributor: str
    publisher: str
    publish_date: Optional[str]
    content: str
    url: str


@dataclass
class CrawlResult:
    """单个 URL 的抓取结果。"""

    url: str
    links: List[str]
    article_data: Optional[ArticleData]
    fetch_ok: bool


class GracefulStopController:
    """监听控制台输入，当用户输入 1 时触发优雅停止。"""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._listener_thread = threading.Thread(target=self._listen_input, daemon=True)

    def start(self) -> None:
        self._listener_thread.start()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def _listen_input(self) -> None:
        print("[控制] 输入 1 并回车，可在当前页面处理完成后安全停止。")
        while not self.stop_event.is_set():
            try:
                cmd = input().strip()
                if cmd == "1":
                    self.stop_event.set()
                    print("[控制] 已收到停止指令，将在当前页面处理完成后退出。")
                    return
                print("[控制] 未识别指令，输入 1 可停止。")
            except EOFError:
                return
            except Exception as exc:
                print(f"[控制] 输入监听异常：{exc}")
                return


class ZZULINewsCrawler:
    """郑州轻工业大学官网新闻爬虫。"""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            }
        )

        self.conn = self._init_database()
        self.crawled_urls = self._load_crawled_urls_from_db()

        self.queue: Deque[str] = deque()
        self.seen_urls: Set[str] = set()
        self._load_or_init_state()

    # ------------------------- 数据库相关 -------------------------

    def _db_connect(self, with_database: bool = True) -> pymysql.connections.Connection:
        kwargs = {
            "host": DB_HOST,
            "port": DB_PORT,
            "user": DB_USER,
            "password": DB_PASSWORD,
            "charset": "utf8mb4",
            "autocommit": False,
        }
        if with_database:
            kwargs["database"] = DB_NAME
        return pymysql.connect(**kwargs)

    def _init_database(self) -> pymysql.connections.Connection:
        """初始化数据库与目标表。"""
        conn_no_db = self._db_connect(with_database=False)
        try:
            with conn_no_db.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                    "DEFAULT CHARACTER SET utf8mb4 "
                    "DEFAULT COLLATE utf8mb4_unicode_ci"
                )
            conn_no_db.commit()
        finally:
            conn_no_db.close()

        conn = self._db_connect(with_database=True)
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{DB_TABLE}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(500),
                    contributor VARCHAR(100),
                    publisher VARCHAR(100),
                    publish_date DATE,
                    content TEXT,
                    url VARCHAR(500) UNIQUE,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
        print(f"[DB] 已确保数据库 `{DB_NAME}` 和数据表 `{DB_TABLE}` 可用。")
        return conn

    def _load_crawled_urls_from_db(self) -> Set[str]:
        """读取已成功入库 URL，用于断点续爬去重。"""
        urls: Set[str] = set()
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"SELECT url FROM `{DB_TABLE}`")
                for row in cursor.fetchall():
                    urls.add(row[0])
        except MySQLError as exc:
            print(f"[DB] 读取历史 URL 失败：{exc}")
        print(f"[DB] 已加载历史成功 URL 数量：{len(urls)}")
        return urls

    def _save_article(self, data: ArticleData) -> bool:
        """写入文章数据，url 唯一冲突时忽略。"""
        sql = (
            f"INSERT INTO `{DB_TABLE}` (title, contributor, publisher, publish_date, content, url) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        data.title[:500],
                        data.contributor[:100],
                        data.publisher[:100],
                        data.publish_date,
                        data.content,
                        data.url[:500],
                    ),
                )
            self.conn.commit()
            self.crawled_urls.add(data.url)
            print(f"[DB] 已入库: {data.url}")
            return True
        except pymysql.err.IntegrityError:
            self.conn.rollback()
            self.crawled_urls.add(data.url)
            print(f"[DB] 重复 URL，跳过插入: {data.url}")
            return False
        except MySQLError as exc:
            self.conn.rollback()
            print(f"[DB] 插入失败: {data.url} | 错误: {exc}")
            return False

    def _save_articles_batch(self, articles: List[ArticleData]) -> int:
        """批量写入文章数据，重复 URL 自动忽略。"""
        if not articles:
            return 0

        sql = (
            f"INSERT IGNORE INTO `{DB_TABLE}` "
            "(title, contributor, publisher, publish_date, content, url) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        payload = [
            (
                item.title[:500],
                item.contributor[:100],
                item.publisher[:100],
                item.publish_date,
                item.content,
                item.url[:500],
            )
            for item in articles
        ]

        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(sql, payload)
                inserted = cursor.rowcount
            self.conn.commit()

            # INSERT IGNORE 的场景下，提交成功表示这些 URL 已处理，可加入去重集合。
            for item in articles:
                self.crawled_urls.add(item.url)

            return inserted
        except MySQLError as exc:
            self.conn.rollback()
            print(f"[DB] 批量插入失败，回退到逐条写入。错误: {exc}")
            inserted = 0
            for item in articles:
                if self._save_article(item):
                    inserted += 1
            return inserted

    # ------------------------- 状态持久化 -------------------------

    def _load_or_init_state(self) -> None:
        """加载断点状态，不存在则初始化。"""
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                pending = state.get("pending_urls", [])
                seen = state.get("seen_urls", [])

                self.queue = deque(self._normalize_urls(pending))
                self.seen_urls = set(self._normalize_urls(seen))

                for url in self.crawled_urls:
                    self.seen_urls.add(url)

                if not self.queue:
                    self._enqueue_seed_url(START_URL)

                print(
                    f"[状态] 已恢复：待爬 {len(self.queue)}，已记录 seen {len(self.seen_urls)}"
                )
                return
            except Exception as exc:
                print(f"[状态] 状态文件损坏或读取失败，将重新初始化。错误: {exc}")

        self.queue = deque()
        self.seen_urls = set(self.crawled_urls)
        self._enqueue_seed_url(START_URL)
        self._save_state()
        print("[状态] 首次启动，已初始化状态文件。")

    def _save_state(self) -> None:
        state = {
            "pending_urls": list(self.queue),
            "seen_urls": list(self.seen_urls),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_urls(urls: List[str]) -> List[str]:
        result: List[str] = []
        for u in urls:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            result.append(u)
        return result

    # ------------------------- URL 与页面处理 -------------------------

    def _is_target_html_url(self, url: str) -> bool:
        """仅允许同域名 + .html/.htm 结尾链接。"""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        domain = parsed.netloc.lower()
        if domain != ALLOWED_DOMAIN and not domain.endswith(f".{ALLOWED_DOMAIN}"):
            return False

        path = parsed.path.lower()
        return path.endswith(".html") or path.endswith(".htm")

    def _enqueue_if_valid(self, url: str) -> None:
        if self._is_target_html_url(url) and url not in self.seen_urls:
            self.queue.append(url)
            self.seen_urls.add(url)

    def _enqueue_seed_url(self, url: str) -> None:
        """加入种子 URL（允许非 .html/.htm），用于首轮链接发现。"""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return
        domain = parsed.netloc.lower()
        if domain != ALLOWED_DOMAIN and not domain.endswith(f".{ALLOWED_DOMAIN}"):
            return
        if url in self.seen_urls:
            return
        self.queue.append(url)
        self.seen_urls.add(url)

    def _extract_links(self, page_url: str, soup: BeautifulSoup) -> List[str]:
        links: List[str] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href:
                continue
            absolute_url = urljoin(page_url, href)
            absolute_url = absolute_url.split("#", 1)[0]
            if self._is_target_html_url(absolute_url):
                links.append(absolute_url)
        return links

    def _fetch_html(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except Exception as exc:
            print(f"[请求] 失败: {url} | 错误: {exc}")
            return None

    def _extract_article_data(self, soup: BeautifulSoup, url: str) -> Optional[ArticleData]:
        article_div = soup.find("div", class_="article")
        if article_div is None:
            return None

        title = self._safe_text(article_div.find("h1", class_="arti-title"))

        contributor_raw = self._safe_text(article_div.find("span", class_="arti-views"))
        contributor = re.sub(r"^\s*供稿单位[:：]\s*", "", contributor_raw)

        publisher_raw = self._safe_text(article_div.find("span", class_="arti-publisher"))
        publisher = re.sub(r"^\s*编辑发布[:：]\s*", "", publisher_raw)

        update_raw = self._safe_text(article_div.find("span", class_="arti-update"))
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", update_raw)
        publish_date = date_match.group(0) if date_match else None

        read_article = article_div.find("article", class_="read")
        paragraphs: List[str] = []
        if read_article is not None:
            for p_tag in read_article.find_all("p"):
                txt = p_tag.get_text(separator=" ", strip=True)
                if txt:
                    paragraphs.append(txt)
        content = "\n\n".join(paragraphs)

        return ArticleData(
            title=title,
            contributor=contributor,
            publisher=publisher,
            publish_date=publish_date,
            content=content,
            url=url,
        )

    @staticmethod
    def _safe_text(tag) -> str:
        if tag is None:
            return ""
        return tag.get_text(separator=" ", strip=True)

    def _process_one_url(self, url: str) -> CrawlResult:
        """抓取并解析单个 URL，供线程池并发调用。"""
        soup = self._fetch_html(url)
        if soup is None:
            return CrawlResult(url=url, links=[], article_data=None, fetch_ok=False)

        links = self._extract_links(url, soup)
        article_data = self._extract_article_data(soup, url)
        return CrawlResult(url=url, links=links, article_data=article_data, fetch_ok=True)

    def _flush_pending_articles(self, pending_articles: List[ArticleData]) -> int:
        """将待写入文章批量落库并清空缓冲。"""
        if not pending_articles:
            return 0
        inserted = self._save_articles_batch(pending_articles)
        pending_articles.clear()
        return inserted

    # ------------------------- 主流程 -------------------------

    def crawl(self) -> None:
        stop_controller = GracefulStopController()
        stop_controller.start()

        processed_count = 0
        stored_count = 0
        skipped_no_article_count = 0
        failed_count = 0
        state_save_round = 0
        pending_articles: List[ArticleData] = []

        print(f"[启动] 初始待爬队列长度: {len(self.queue)}")
        print(f"[启动] 开始并发 BFS 爬取，线程数: {MAX_WORKERS}\n")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while self.queue:
                if stop_controller.should_stop():
                    print("[停止] 用户请求停止，将在当前批次处理完成后退出。")
                    break

                batch_urls: List[str] = []
                batch_limit = MAX_WORKERS * 2
                while self.queue and len(batch_urls) < batch_limit:
                    current_url = self.queue.popleft()
                    if current_url in self.crawled_urls:
                        continue
                    batch_urls.append(current_url)

                if not batch_urls:
                    continue

                futures = [executor.submit(self._process_one_url, url) for url in batch_urls]

                newly_queued_total = 0
                batch_no_article_count = 0
                batch_failed_count = 0

                for future in as_completed(futures):
                    result = future.result()
                    processed_count += 1

                    if not result.fetch_ok:
                        batch_failed_count += 1
                        continue

                    for link in result.links:
                        before = len(self.queue)
                        self._enqueue_if_valid(link)
                        if len(self.queue) > before:
                            newly_queued_total += 1

                    if result.article_data is None:
                        batch_no_article_count += 1
                    else:
                        pending_articles.append(result.article_data)
                        if len(pending_articles) >= DB_BATCH_SIZE:
                            stored_count += self._flush_pending_articles(pending_articles)

                skipped_no_article_count += batch_no_article_count
                failed_count += batch_failed_count

                if pending_articles and (
                    stop_controller.should_stop() or len(pending_articles) >= DB_BATCH_SIZE
                ):
                    stored_count += self._flush_pending_articles(pending_articles)

                state_save_round += 1
                if state_save_round % STATE_SAVE_EVERY_BATCHES == 0:
                    if pending_articles:
                        stored_count += self._flush_pending_articles(pending_articles)
                    self._save_state()

                print(
                    f"[批次] 本批处理: {len(batch_urls)} | 失败: {batch_failed_count} | "
                    f"无 article: {batch_no_article_count} | 新增链接: {newly_queued_total} | "
                    f"待爬剩余: {len(self.queue)}"
                )

                if REQUEST_INTERVAL_SECONDS > 0:
                    time.sleep(REQUEST_INTERVAL_SECONDS)

        if pending_articles:
            stored_count += self._flush_pending_articles(pending_articles)

        self._save_state()
        self.close()

        print("\n================ 爬取结束 ================")
        print(f"处理页面总数: {processed_count}")
        print(f"成功入库数量: {stored_count}")
        print(f"因无 div.article 跳过数量: {skipped_no_article_count}")
        print(f"请求失败数量: {failed_count}")
        print(f"当前待爬队列长度: {len(self.queue)}")
        print(f"状态文件位置: {STATE_FILE}")
        print("=========================================")

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def main() -> None:
    crawler = ZZULINewsCrawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
