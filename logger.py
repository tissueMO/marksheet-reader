# coding: utf-8
###############################################################################
#    処理時間を計測するモジュールです。
###############################################################################
import os
import pytz
import logging
from datetime import datetime as dt
from logging import config


# ログ出力設定ロード
config.fileConfig("logging.conf")


class Logger():
    """ログ出力機構を担うクラスです。
    """

    # ロガーの名前
    LOGGER_NAME = "MarksheetReader"

    def __init__(self, module_name: str = None):
        """コンストラクター

        Arguments:
            module_name {str} -- モジュール名
        """
        self.module_name = module_name

        # ログ出力オブジェクト
        self._logger = logging.getLogger(Logger.LOGGER_NAME)

    def _create_log_text(
            self, code: str = "", level_prefix: str = "", message: str = ""):
        """ログメッセージを一定のフォーマットに従って生成します。

        Arguments:
            code {str} -- メッセージコード
            level_prefix {str} -- ログ種別のプレフィックス
            message {str} -- メッセージ本文
        """
        # タイムゾーンを日本にして時刻を取得
        now = dt.now(pytz.timezone("Asia/Tokyo"))
        time = now.strftime("%Y-%m-%d %H:%M:%S.") + \
            f"{now.microsecond // 1000:03d}"

        # メッセージ生成
        message = f"{time} [{level_prefix}] {self.module_name}: {message}"

        return message

    def log_debug(self, message: str):
        """デバッグログを出力します。

        Arguments:
            message {str} -- メッセージ内容
        """
        self._logger.debug(
            self._create_log_text(level_prefix="D", message=message)
        )

    def log_info(self, message: str):
        """情報ログを出力します。

        Arguments:
            message {str} -- メッセージ内容
        """
        self._logger.info(
            self._create_log_text(level_prefix="I", message=message)
        )

    def log_warn(self, message: str):
        """警告ログを出力します。

        Arguments:
            message {str} -- メッセージ内容
        """
        self._logger.warning(
            self._create_log_text(level_prefix="W", message=message)
        )

    def log_error(self, message: str):
        """エラーログを出力します。

        Arguments:
            message {str} -- メッセージ内容
        """
        self._logger.error(
            self._create_log_text(level_prefix="E", message=message)
        )

    def log_critical(self, message: str):
        """致命的エラーログを出力します。

        Arguments:
            code {str} -- メッセージコード
            message {str} -- メッセージ内容
        """
        self.logger.critical(
            self._create_log_text(level_prefix="C", message=message)
        )
