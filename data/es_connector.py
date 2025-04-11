# data/es_connector.py
import logging
import json
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)


class ESConnector:
    """Elasticsearch连接器，用于从ES中读取日志数据"""

    def __init__(
            self,
            es_host: str = "localhost",
            es_port: int = 9200,
            es_user: Optional[str] = None,
            es_password: Optional[str] = None,
            es_index_pattern: str = "logstash-*",
            use_ssl: bool = False,
            verify_certs: bool = False
    ):
        """
        初始化ES连接器

        Args:
            es_host: ES主机地址
            es_port: ES端口
            es_user: ES用户名（可选）
            es_password: ES密码（可选）
            es_index_pattern: 要查询的索引模式
            use_ssl: 是否使用SSL连接
            verify_certs: 是否验证SSL证书
        """
        # 初始化ES客户端
        self.es_config = {
            "host": es_host,
            "port": es_port,
            "use_ssl": use_ssl,
            "verify_certs": verify_certs
        }

        if es_user and es_password:
            self.es_config["http_auth"] = (es_user, es_password)

        self.es_client = Elasticsearch([self.es_config])
        self.es_index_pattern = es_index_pattern

        # 检查ES连接
        if not self.es_client.ping():
            raise ConnectionError(f"无法连接到 Elasticsearch: {es_host}:{es_port}")

        logger.info(f"成功连接到 Elasticsearch: {es_host}:{es_port}")

        # 状态跟踪
        self.last_processed_timestamp = None
        self.batch_size = 5000

    def extract_logs(
            self,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            max_logs: int = 100000,
            query_filter: Optional[Dict] = None,
            fields: Optional[List[str]] = None,
            sort_field: str = "@timestamp"
    ) -> pd.DataFrame:
        """
        从ES中提取日志数据

        Args:
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            max_logs: 最大提取日志数
            query_filter: 额外的查询过滤条件
            fields: 要提取的字段列表（如果为None则提取所有字段）
            sort_field: 排序字段

        Returns:
            包含日志数据的DataFrame
        """
        # 构建查询
        query = {"match_all": {}}

        if start_time or end_time or query_filter:
            query = {"bool": {"must": []}}

            # 添加时间范围过滤
            if start_time or end_time:
                time_range = {}
                if start_time:
                    time_range["gte"] = start_time.isoformat()
                if end_time:
                    time_range["lte"] = end_time.isoformat()

                query["bool"]["must"].append({"range": {"@timestamp": time_range}})

            # 添加其他过滤条件
            if query_filter:
                if isinstance(query_filter, dict):
                    query["bool"]["must"].append(query_filter)
                else:
                    query["bool"]["must"].extend(query_filter)

        # 准备搜索请求
        search_body = {
            "query": query,
            "sort": [{sort_field: {"order": "asc"}}]
        }

        # 如果指定了字段，则只提取这些字段
        if fields:
            search_body["_source"] = fields

        # 使用ES scroll API批量获取数据
        logger.info(f"开始从ES提取日志，最大提取数量: {max_logs}")

        results = []
        try:
            # 初始化scroll
            scroll_resp = self.es_client.search(
                index=self.es_index_pattern,
                body=search_body,
                scroll="5m",
                size=self.batch_size
            )

            # 获取第一批结果
            scroll_id = scroll_resp["_scroll_id"]
            hits = scroll_resp["hits"]["hits"]
            total_fetched = len(hits)

            while hits and total_fetched <= max_logs:
                # 处理当前批次
                for hit in hits:
                    # 提取完整的源数据
                    source_data = hit["_source"]

                    # 添加ES元数据
                    record = {
                        "_id": hit["_id"],
                        "_index": hit["_index"],
                        "_score": hit.get("_score")
                    }

                    # 合并源数据
                    record.update(source_data)

                    results.append(record)

                # 如果已经达到最大数量，则退出
                if total_fetched >= max_logs:
                    break

                # 获取下一批结果
                scroll_resp = self.es_client.scroll(scroll_id=scroll_id, scroll="5m")
                scroll_id = scroll_resp["_scroll_id"]
                hits = scroll_resp["hits"]["hits"]
                total_fetched += len(hits)

                logger.info(f"已提取 {total_fetched} 条日志记录")

            # 清理scroll
            self.es_client.clear_scroll(body={"scroll_id": scroll_id})

        except Exception as e:
            logger.error(f"提取日志时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # 转换为DataFrame
        df = pd.DataFrame(results)

        if not df.empty and "@timestamp" in df.columns:
            # 更新最后处理的时间戳
            self.last_processed_timestamp = df["@timestamp"].max()

        logger.info(f"成功从ES提取 {len(df)} 条日志记录")
        return df

    def save_to_new_index(
            self,
            df: pd.DataFrame,
            target_index: str,
            id_column: str = "_id",
            batch_size: int = 1000
    ) -> Tuple[int, int]:
        """
        将DataFrame数据保存到新的ES索引

        Args:
            df: 要保存的DataFrame
            target_index: 目标索引名称
            id_column: 作为文档ID的列名
            batch_size: 批处理大小

        Returns:
            (总记录数, 成功保存数)
        """
        if df.empty:
            logger.warning("没有数据需要保存到ES")
            return 0, 0

        # 检查目标索引是否存在，如果不存在则创建
        if not self.es_client.indices.exists(index=target_index):
            self.es_client.indices.create(
                index=target_index,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "ik_smart_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "ik_smart"
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "message": {
                                "type": "text",
                                "analyzer": "ik_smart_analyzer"
                            },
                            "@timestamp": {
                                "type": "date"
                            }
                        }
                    }
                }
            )
            logger.info(f"创建索引 {target_index} 成功")

        # 批量保存数据
        total_records = len(df)
        successful_records = 0

        try:
            actions = []

            for _, row in df.iterrows():
                # 将行数据转换为字典
                doc = row.to_dict()

                # 提取ID
                doc_id = None
                if id_column in doc:
                    doc_id = doc[id_column]
                    # 可选：从文档中移除ID字段
                    # del doc[id_column]

                # 创建操作
                action = {
                    "_index": target_index,
                    "_source": doc
                }

                if doc_id:
                    action["_id"] = doc_id

                actions.append(action)

                # 批量处理
                if len(actions) >= batch_size:
                    success, failed = helpers.bulk(
                        self.es_client,
                        actions,
                        stats_only=True,
                        raise_on_error=False
                    )
                    successful_records += success
                    actions = []

            # 处理剩余的记录
            if actions:
                success, failed = helpers.bulk(
                    self.es_client,
                    actions,
                    stats_only=True,
                    raise_on_error=False
                )
                successful_records += success

            logger.info(f"成功保存 {successful_records}/{total_records} 条记录到索引 {target_index}")
            return total_records, successful_records

        except Exception as e:
            logger.error(f"保存数据到ES时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return total_records, successful_records