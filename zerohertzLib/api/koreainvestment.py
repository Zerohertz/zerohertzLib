"""
MIT License for Original Code (sharebook-kr, https://github.com/sharebook-kr/mojito)

Copyright (c) 2022 sharebook-kr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MIT License for Modified Code (Hyogeun Oh)

Copyright (c) 2023 Hyogeun Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import datetime
import json
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


class KoreaInvestment:
    """한국투자증권 API를 호출하는 class

    Note:
        `공식 API 문서 <https://apiportal.koreainvestment.com/apiservice>`_ 및 `mojito <https://github.com/sharebook-kr/mojito>`_ 참고

        ``secret.key`` 는 아래와 같이 구성

        .. code-block::

            zero... (API Key)
            hertz... (API Secret)

    Args:
        account_no (``str``): API 호출 시 사용할 계좌 번호
        path (``Optional[str]``): ``secret.key`` 혹은 ``token.dat`` 이 포함된 경로

    Attributes:
        api_key (``str``): API Key
        api_secret (``str``): API Secret

    Examples:
        >>> broker = zz.api.KoreaInvestment("00000000-00")
    """

    def __init__(
        self,
        account_no: str,
        path: Optional[str] = "./",
    ) -> None:
        self.account_no = account_no
        self.account_no_prefix = self.account_no.split("-")[0]
        self.account_no_postfix = self.account_no.split("-")[1]
        self.path = path
        self.base_url = "https://openapi.koreainvestment.com:9443"
        files = os.listdir(path)
        if "secret.key" in files:
            self._load_secret()
            if self._check_token():
                self._load_token()
            else:
                self._issue_token()
        else:
            raise FileNotFoundError(
                f"Required files ('secret.key') not found in {path}"
            )

    def _load_secret(self) -> None:
        """``secret.key`` load

        Returns:
            ``None``: Attribute에 API 속성들 추가
        """
        with open(os.path.join(self.path, "secret.key"), "r", encoding="utf-8") as file:
            secrets = file.readlines()
        self.api_key = secrets[0].strip()
        self.api_secret = secrets[1].strip()

    def _check_token(self) -> bool:
        """``token.dat`` check

        Returns:
            ``bool``: ``token.dat`` 존재 및 유효성 여부
        """
        try:
            with open(os.path.join(self.path, "token.dat"), "rb") as file:
                data = pickle.load(file)
            expire_epoch = data["timestamp"]
            now_epoch = int(datetime.datetime.now().timestamp())
            status = False
            if (
                (now_epoch - expire_epoch > 0)
                or (data["api_key"] != self.api_key)
                or (data["api_secret"] != self.api_secret)
            ):
                status = False
            else:
                status = True
            return status
        except IOError:
            return False

    def _issue_token(self) -> None:
        """``token.dat`` 발급

        Returns:
            ``None``: ``token.dat`` 발급 후 attribute에 추가
        """
        print("=" * 100)
        print(
            "[오픈API 서비스 안내]\n\n고객님 명의의 오픈API 접근 토큰이 발급되었습니다.\n* 접근 토큰 유효기간: 발급시점부터 24시간\n\n※ 유의사항\n\t1. 고객 본인께서 인지하신 정상 발급인지 확인 바랍니다.\n\t2. 정상 발급이 아닌 경우에는 즉시 해지(계좌별)하신 후 재 신청 하시기 바랍니다.\n\t당사홈페이지 > 트레이딩 > Open API > KIS Developers > KIS Developers 신청하기 메뉴를 통해서 해지(계좌별) 후 재 신청(추가신청하기)\n\t3. 접근 토큰은 1일 1회 발급 원칙이며, 유효기간내 잦은 토큰 발급 발생 시 이용이 제한 될 수 있습니다.\n\n※고객센터 상담원 연결\n☎1544-5000"
        )
        print("=" * 100)
        path = "oauth2/tokenP"
        url = f"{self.base_url}/{path}"
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
        }
        response = requests.post(
            url, headers=headers, data=json.dumps(data), timeout=10
        ).json()
        self.access_token = f"Bearer {response['access_token']}"
        now = datetime.datetime.now()
        response["timestamp"] = int(now.timestamp()) + response["expires_in"] - 600
        response["api_key"] = self.api_key
        response["api_secret"] = self.api_secret
        with open(os.path.join(self.path, "token.dat"), "wb") as file:
            pickle.dump(response, file)

    def _load_token(self) -> None:
        """``token.dat`` load

        Returns:
            ``None``: ``token.dat`` attribute에 추가
        """
        with open(os.path.join(self.path, "token.dat"), "rb") as file:
            data = pickle.load(file)
            self.access_token = f"Bearer {data['access_token']}"

    def get_price(self, symbol: str, kor: Optional[bool] = True) -> Dict[str, Dict]:
        """주식 현재가 시세

        Args:
            symbol (``str``): 종목 code
            kor (``Optional[bool]``): 국내 여부

        Returns:
            ``Dict[str, Dict]``: 주식 현재가 시세

        Examples:
            >>> samsung = broker.get_price("005930")
            >>> samsung["output"]["stck_prpr"]   # 주식 현재가
            >>> samsung["output"]["per"]         # PER (Price-to-Earnings Ratio, 주가수익비율)
            >>> samsung["output"]["pbr"]         # PBR (Price-to-Book Ratio, 주가순자산비율)
            >>> samsung["output"]["eps"]         # EPS (Earnings Per Share, 주당순이익)
            >>> samsung["output"]["bps"]         # BPS (Book-value Per Share, 주당순자산가치)
            >>> samsung["output"]["w52_hgpr"]    # 52주일 최고가
            >>> samsung["output"]["w52_lwpr"]    # 52주일 최저가
            >>> apple = broker.get_price("AAPL", kor=False)
            >>> apple["output"]["last"]          # 현재가
            >>> apple["output"]["ordy"]          # 매수가능여부
        """
        if kor:
            return self._get_korea_price(symbol)
        return self._get_oversea_price(symbol)

    def _get_korea_price(self, symbol: str) -> Dict[str, Dict]:
        """주식 현재가 시세 [v1_국내주식-008]

        Args:
            symbol (``str``): 종목 code

        Returns:
            ``Dict[str, Dict]``: 국내 주식 현재가 시세
        """
        path = "uapi/domestic-stock/v1/quotations/inquire-price"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": self.access_token,
            "appKey": self.api_key,
            "appSecret": self.api_secret,
            "tr_id": "FHKST01010100",
        }
        params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return response.json()

    def _get_oversea_price(self, symbol: str) -> Dict[str, Dict]:
        """해외 주식 현재체결가 [v1_해외주식-009]

        해외주식 시세는 무료시세(지연체결가)만이 제공되며, API로는 유료시세(실시간체결가)를 받아보실 수 없습니다.

        Args:
            symbol (``str``): 종목 code

        Returns:
            ``Dict[str, Dict]``: 해외 주식 현재가 시세
        """
        path = "uapi/overseas-price/v1/quotations/price"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": self.access_token,
            "appKey": self.api_key,
            "appSecret": self.api_secret,
            "tr_id": "HHDFS00000300",
        }
        params = {"AUTH": "", "EXCD": "NAS", "SYMB": symbol}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return response.json()

    def get_ohlcv(
        self,
        symbol: str,
        time_frame: Optional[str] = "D",
        start_day: Optional[str] = "",
        end_day: Optional[str] = "",
        adj_price: Optional[bool] = True,
        kor: Optional[bool] = True,
    ) -> Dict[str, Dict]:
        """종목 code에 따른 기간별 OHLCV (Open, High, Low, Close, Volume)

        Args:
            symbol (``str``): 종목 code
            time_frame (``Optional[str]``): 시간 window size (``"D"``: 일, ``"W"``: 주, ``"M"``: 월, ``"Y"``: 년)
            start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
            end_day (``Optional[str]``): 조회 종료 일자 (``YYYYMMDD``)
            adj_price (``Optional[bool]``): 수정 주가 반영 여부
            kor (``Optional[bool]``): 국내 여부

        Returns:
            ``Dict[str, Dict]``: OHLCV (Open, High, Low, Close, Volume)

        Examples:
            >>> broker.get_ohlcv("005930")
            {'output1': {'prdy_vrss': '...', ...}, 'output2': ['stck_bsop_date': '...', ...]}
            >>> broker.get_ohlcv("AAPL", kor=False)
            {'output1': {'rsym': '...', ...}, 'output2': ['xymd': '...', ...]}
        """
        if kor:
            return self._get_korea_ohlcv(
                symbol, time_frame, start_day, end_day, adj_price
            )
        return self._get_oversea_ohlcv(
            symbol, time_frame, start_day, end_day, adj_price
        )

    def _get_korea_ohlcv(
        self,
        symbol: str,
        time_frame: Optional[str] = "D",
        start_day: Optional[str] = "",
        end_day: Optional[str] = "",
        adj_price: Optional[bool] = True,
    ) -> Dict[str, Dict]:
        """국내 주식 기간별 시세 (일/주/월/년) [v1_국내주식-016]

        한 번의 호출에 최대 100건까지 확인 가능합니다.

        Args:
            symbol (``str``): 종목 code
            time_frame (``Optional[str]``): 시간 window size (``"D"``: 일, ``"W"``: 주, ``"M"``: 월, ``"Y"``: 년)
            start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
            end_day (``Optional[str]``): 조회 종료 일자 (``YYYYMMDD``)
            adj_price (``Optional[bool]``): 수정 주가 반영 여부

        Returns:
            ``Dict[str, Dict]``: 국내 주식의 기간별 시세
        """
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": self.access_token,
            "appKey": self.api_key,
            "appSecret": self.api_secret,
            "tr_id": "FHKST03010100",
        }
        if end_day == "":
            now = datetime.datetime.now()
            end_day = now.strftime("%Y%m%d")
        if start_day == "":
            start_day = "19800104"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": start_day,
            "FID_INPUT_DATE_2": end_day,
            "FID_PERIOD_DIV_CODE": time_frame,
            "FID_ORG_ADJ_PRC": 0 if adj_price else 1,
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        if not start_day == "19800104":
            while (
                "stck_bsop_date" in data["output2"][-1].keys()
                and start_day < data["output2"][-1]["stck_bsop_date"]
            ):
                params["FID_INPUT_DATE_2"] = data["output2"][-1]["stck_bsop_date"]
                response = requests.get(url, headers=headers, params=params, timeout=10)
                data_ = response.json()
                if (
                    "stck_bsop_date" not in data_["output2"][-1].keys()
                    or data["output2"][-1]["stck_bsop_date"]
                    == data_["output2"][-1]["stck_bsop_date"]
                ):
                    break
                data["output2"] += data_["output2"][1:]
                time.sleep(0.02)
        return data

    def _get_oversea_ohlcv(
        self,
        symbol: str,
        time_frame: Optional[str] = "D",
        start_day: Optional[str] = "",
        end_day: Optional[str] = "",
        adj_price: Optional[bool] = True,
    ) -> Dict[str, Dict]:
        """해외 주식 기간별 시세 [v1_해외주식-010]

        한 번의 호출에 최대 100건까지 확인 가능합니다.
        해외주식 시세는 무료시세 (지연체결가)만이 제공되며, API로는 유료시세 (실시간체결가)를 받아보실 수 없습니다.

        Args:
            symbol (``str``): 종목 code
            time_frame (``Optional[str]``): 시간 window size (``"D"``: 일, ``"W"``: 주, ``"M"``: 월)
            start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
            end_day (``Optional[str]``): 조회 종료 일자 (``YYYYMMDD``)
            adj_price (``Optional[bool]``): 수정 주가 반영 여부

        Returns:
            ``Dict[str, Dict]``: 해외 주식의 기간별 시세
        """
        path = "uapi/overseas-price/v1/quotations/dailyprice"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": self.access_token,
            "appKey": self.api_key,
            "appSecret": self.api_secret,
            "tr_id": "HHDFS76240000",
        }
        timeframe_lookup = {"D": "0", "W": "1", "M": "2"}
        if end_day == "":
            now = datetime.datetime.now()
            end_day = now.strftime("%Y%m%d")
        params = {
            "AUTH": "",
            "EXCD": "NAS",
            "SYMB": symbol,
            "GUBN": timeframe_lookup.get(time_frame, "0"),
            "BYMD": end_day,
            "MODP": 1 if adj_price else 0,
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        if start_day != "":
            while start_day < data["output2"][-1]["xymd"]:
                params["BYMD"] = data["output2"][-1]["xymd"]
                response = requests.get(url, headers=headers, params=params, timeout=10)
                data_ = response.json()
                if (
                    "xymd" not in data_["output2"][-1].keys()
                    or data["output2"][-1]["xymd"] == data_["output2"][-1]["xymd"]
                ):
                    break
                data["output2"] += data_["output2"][1:]
                time.sleep(0.02)
        return data

    def response2ohlcv(self, response: Dict[str, Dict]) -> Tuple[str, pd.DataFrame]:
        """``get_ohlcv`` 에 의한 응답을 ``pd.DataFrame`` 으로 변환

        Args:
            response (``Dict[str, Dict]``): ``get_ohlcv`` 의 출력

        Returns:
            ``Tuple[str, pd.DataFrame]``: 종목의 이름과 OHLCV (Open, High, Low, Close, Volume)

        Examples:
            >>> samsung = broker.get_ohlcv("005930")
            >>> broker.response2ohlcv(samsung)
            ('삼성전자',  Open    High     Low      Close     Volume
            2023-07-10  70000.0  70400.0  69200.0  69500.0  11713926.0
            ...             ...      ...      ...      ...         ...
            2023-12-04  72800.0  72900.0  72400.0  72700.0   7917006.0
            [100 rows x 5 columns])
            >>> apple = broker.get_ohlcv("AAPL", kor=False)
            >>> broker.response2ohlcv(apple)
            ('AAPL',    Open      High      Low       Close     Volume
            2023-07-13  189.9927  190.6808  189.2746  190.0325  41342338.0
            ...              ...       ...       ...       ...         ...
            2023-12-01  190.3300  191.5600  189.2300  191.2400  45704823.0
            [100 rows x 5 columns])
        """
        date = []
        data = defaultdict(list)
        if "rsym" in response["output1"].keys():
            name = response["output1"]["rsym"][4:]
            date_key, open_key, high_key, low_key, close_key, volume_key = (
                "xymd",
                "open",
                "high",
                "low",
                "clos",
                "tvol",
            )
        else:
            name = response["output1"]["hts_kor_isnm"]
            date_key, open_key, high_key, low_key, close_key, volume_key = (
                "stck_bsop_date",
                "stck_oprc",
                "stck_hgpr",
                "stck_lwpr",
                "stck_clpr",
                "acml_vol",
            )
        for data_ in response["output2"]:
            try:
                date.append(data_[date_key])
                data["Open"].append(float(data_[open_key]))
                data["High"].append(float(data_[high_key]))
                data["Low"].append(float(data_[low_key]))
                data["Close"].append(float(data_[close_key]))
                data["Volume"].append(float(data_[volume_key]))
            except KeyError as error:
                if date:
                    return (
                        name,
                        pd.DataFrame(data, index=pd.to_datetime(date, format="%Y%m%d"))[
                            ::-1
                        ],
                    )
                raise error
        return (
            name,
            pd.DataFrame(data, index=pd.to_datetime(date, format="%Y%m%d"))[::-1],
        )

    def get_ohlcvs(
        self,
        symbols: List[str],
        time_frame: Optional[str] = "D",
        start_day: Optional[str] = "",
        end_day: Optional[str] = "",
        adj_price: Optional[bool] = True,
        kor: Optional[bool] = True,
    ) -> Tuple[List[str], List[pd.DataFrame]]:
        """여러 종목 code에 따른 기간별 OHLCV (Open, High, Low, Close, Volume)

        Args:
            symbols (``List[str]``): 종목 code들
            time_frame (``Optional[str]``): 시간 window size (``"D"``: 일, ``"W"``: 주, ``"M"``: 월, ``"Y"``: 년)
            start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
            end_day (``Optional[str]``): 조회 종료 일자 (``YYYYMMDD``)
            adj_price (``Optional[bool]``): 수정 주가 반영 여부
            kor (``Optional[bool]``): 국내 여부

        Returns:
            ``Tuple[List[str], List[pd.DataFrame]]``: Code들에 따른 종목의 이름과 OHLCV (Open, High, Low, Close, Volume)

        Examples:
            >>> broker.get_ohlcvs(["005930", "035420"], start_day="20221205")
            (['삼성전자', 'NAVER'],
            [               Open     High      Low    Close      Volume
            2022-12-05  60900.0  61100.0  60000.0  60300.0  13767787.0
            ...             ...      ...      ...      ...         ...
            2023-12-05  72300.0  72400.0  71500.0  71500.0   4598639.0
            [248 rows x 5 columns],
                            Open      High       Low     Close     Volume
            2022-12-05  187000.0  195000.0  186500.0  191500.0  1224361.0
            ...              ...       ...       ...       ...        ...
            2023-12-05  210000.0  216500.0  209500.0  213500.0   454184.0
            [248 rows x 5 columns]])
        """
        title = []
        data = []
        for symbol in symbols:
            try:
                response = self.get_ohlcv(
                    symbol, time_frame, start_day, end_day, adj_price, kor
                )
                title_, data_ = self.response2ohlcv(response)
                title.append(title_)
                data.append(data_)
            except KeyError:
                print(f"'{symbol}' is not found")
        return title, data

    def get_balance(self, kor: Optional[bool] = True) -> Dict[str, Dict]:
        """주식 계좌 잔고 조회

        Args:
            kor (``Optional[bool]``): 국내 여부

        Returns:
            ``Dict[str, Dict]``: 계좌 내역

        Examples:
            >>> balance = broker.get_balance()
            {'output1': [{'pdno': '...', ...}], 'output2': [{'dnca_tot_amt': '...', ...}]}
            >>> balance["output1"][0]["pdno"]                # 종목번호 (뒷 6자리)
            >>> balance["output1"][0]["prdt_name"]           # 종목명
            >>> balance["output1"][0]["hldg_qty"]            # 보유수량
            >>> balance["output1"][0]["pchs_avg_pric"]       # 매입평균가격 (매입금액 / 보유수량)
            >>> balance["output1"][0]["pchs_amt"]            # 매입금액
            >>> balance["output1"][0]["prpr"]                # 현재가
            >>> balance["output1"][0]["evlu_amt"]            # 평가금액
            >>> balance["output1"][0]["evlu_pfls_amt"]       # 평가손익금액 (평가금액 - 매입금액)
            >>> balance["output1"][0]["evlu_pfls_rt"]        # 평가손익율
            >>> balance["output1"][0]["evlu_erng_rt"]        # 평가수익율
            >>> balance["output2"][0]["dnca_tot_amt"]        # 예수금총금액
            >>> balance["output2"][0]["thdt_buy_amt"]        # 금일매수금액
            >>> balance["output2"][0]["tot_evlu_amt"]        # 총평가금액 (유가증권 평가금액 합계금액 + D+2 예수금)
            >>> balance["output2"][0]["nass_amt"]            # 순자산금액
            >>> balance["output2"][0]["pchs_amt_smtl_amt"]   # 매입금액합계금액
            >>> balance["output2"][0]["evlu_amt_smtl_amt"]   # 평가금액합계금액
            >>> balance["output2"][0]["evlu_pfls_smtl_amt"]  # 평가손익합계금액
            >>> broker.get_balance(False)
            {'output1': [], 'output2': {'frcr_pchs_amt1': '...', ...}}
            {'output1': [{'prdt_name': '...', ...}], 'output2': [{'crcy_cd': 'USD', ...}], 'output3': {'pchs_amt_smtl': '...', ...}}
        """
        if kor:
            output = {}
            data = self._get_korea_balance()
            output["output1"] = data["output1"]
            output["output2"] = data["output2"]
            while data["tr_cont"] == "M":
                fk100 = data["ctx_area_fk100"]
                nk100 = data["ctx_area_nk100"]
                data = self._get_korea_balance(fk100, nk100)
                output["output1"].extend(data["output1"])
                output["output2"].extend(data["output2"])
            return output
        return self._get_oversea_balance()

    def _get_korea_balance(
        self, ctx_area_fk100: Optional[str] = "", ctx_area_nk100: Optional[str] = ""
    ) -> Dict[str, Dict]:
        """주식 잔고 조회 [v1_국내주식-006]

        실전계좌의 경우, 한 번의 호출에 최대 50건까지 확인 가능하며, 이후의 값은 연속조회를 통해 확인하실 수 있습니다.

        Args:
            ctx_area_fk100 (``Optional[str]``): 연속조회검색조건100
            ctx_areak_nk100 (``Optional[str]``): 연속조회키100

        Returns:
            ``Dict[str, Dict]``: 잔고 조회 결과
        """
        path = "uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": self.access_token,
            "appKey": self.api_key,
            "appSecret": self.api_secret,
            "tr_id": "TTTC8434R",
        }
        params = {
            "CANO": self.account_no_prefix,
            "ACNT_PRDT_CD": self.account_no_postfix,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": ctx_area_fk100,
            "CTX_AREA_NK100": ctx_area_nk100,
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        data["tr_cont"] = response.headers["tr_cont"]
        return data

    # def _get_oversea_balance(
    #     self, ctx_area_fk200: Optional[str] = "", ctx_area_nk200: Optional[str] = ""
    # ) -> Dict[str, Dict]:
    #     """해외 주식 잔고 [v1_해외주식-006]
    #
    #     실전계좌의 경우, 한 번의 호출에 최대 100건까지 확인 가능하며, 이후의 값은 연속조회를 통해 확인하실 수 있습니다.
    #
    #     Args:
    #         ctx_area_fk200 (``Optional[str]``): 연속조회검색조건200
    #         ctx_area_nk200 (``Optional[str]``): 연속조회키200
    #
    #     Returns:
    #         ``Dict[str, Dict]``: 잔고 조회 결과
    #     """
    #     path = "uapi/overseas-stock/v1/trading/inquire-balance"
    #     url = f"{self.base_url}/{path}"
    #     headers = {
    #         "content-type": "application/json",
    #         "authorization": self.access_token,
    #         "appKey": self.api_key,
    #         "appSecret": self.api_secret,
    #         "tr_id": "JTTT3012R",  # TTTS3012R
    #     }
    #     params = {
    #         "CANO": self.account_no_prefix,
    #         "ACNT_PRDT_CD": self.account_no_postfix,
    #         "OVRS_EXCG_CD": "NASD",
    #         "TR_CRCY_CD": "USD",
    #         "CTX_AREA_FK200": ctx_area_fk200,
    #         "CTX_AREA_NK200": ctx_area_nk200,
    #     }
    #     response = requests.get(url, headers=headers, params=params, timeout=10)
    #     data = response.json()
    #     data["tr_cont"] = response.headers["tr_cont"]
    #     return data

    def _get_oversea_balance(self) -> Dict[str, Dict]:
        """해외 주식 체결기준현재잔고[v1_해외주식-008]

        Returns:
            ``Dict[str, Dict]``: 잔고 조회 결과
        """
        path = "uapi/overseas-stock/v1/trading/inquire-present-balance"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": self.access_token,
            "appKey": self.api_key,
            "appSecret": self.api_secret,
            "tr_id": "CTRP6504R",
        }
        params = {
            "CANO": self.account_no_prefix,
            "ACNT_PRDT_CD": self.account_no_postfix,
            "WCRC_FRCR_DVSN_CD": "02",
            "NATN_CD": "840",
            "TR_MKET_CD": "00",
            "INQR_DVSN_CD": "00",
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        return data

    def get_conclusion(self) -> Dict[str, Dict]:
        """주식 계좌 잔고의 국내 실현손익 조회

        Returns:
            ``Dict[str, Dict]``: 잔고 실현손익 조회 결과

        Examples:
            >>> conclusion = broker.get_conclusion()
        """
        output = {}
        data = self._get_conclusion()
        output["output1"] = data["output1"]
        output["output2"] = data["output2"]
        while data["tr_cont"] == "M":
            fk100 = data["ctx_area_fk100"]
            nk100 = data["ctx_area_nk100"]
            data = self._get_conclusion(fk100, nk100)
            output["output1"].extend(data["output1"])
            output["output2"].extend(data["output2"])
        return output

    def _get_conclusion(
        self, ctx_area_fk100: Optional[str] = "", ctx_area_nk100: Optional[str] = ""
    ) -> Dict[str, Dict]:
        """주식 잔고 조회 실현손익 [v1_국내주식-041]

        Args:
            ctx_area_fk100 (``Optional[str]``): 연속조회검색조건100
            ctx_areak_nk100 (``Optional[str]``): 연속조회키100

        Returns:
            ``Dict[str, Dict]``: 잔고 실현손익 조회 결과
        """
        path = "uapi/domestic-stock/v1/trading/inquire-balance-rlz-pl"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": self.access_token,
            "appKey": self.api_key,
            "appSecret": self.api_secret,
            "tr_id": "TTTC8494R",
        }
        params = {
            "CANO": self.account_no_prefix,
            "ACNT_PRDT_CD": self.account_no_postfix,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "COST_ICLD_YN": "",
            "CTX_AREA_FK100": ctx_area_fk100,
            "CTX_AREA_NK100": ctx_area_nk100,
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        data["tr_cont"] = response.headers["tr_cont"]
        return data
