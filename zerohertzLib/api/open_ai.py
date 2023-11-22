"""
MIT License

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

from typing import Optional

import openai


class OpenAI(openai.OpenAI):
    """OpenAI의 client instance 생성

    Note:
        `공식 OpenAI GitHub <https://github.com/openai/openai-python>`_ 참고

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280564436-27d62c06-f59a-4772-8cd6-48be49147a86.png
        :alt: OpenAI API Key
        :align: center
        :width: 300px

    위와 같이 `OpenAI <https://platform.openai.com/account/api-keys>`_ page에서 발급 후 API를 등록해야 사용할 수 있다.

    Args:
        api_key (``str``): 위에서 등록한 OpenAI의 API key

    Attributes:
        model (``List[str]``): 사용 가능한 model의 이름

    Methods:
        __call__:
            Model 호출 수행

            Args:
                message (``str``): Model 호출 시 사용될 입력
                model (``str``): 호출할 model 선택
                stream (``str``): 응답의 실시간 출력 여부

            Returns:
                ``str``: 호출된 model의 결과

    Examples:
        >>> client = zz.api.OpenAI("sk-...")
        >>> client.models.list()
        SyncPage[Model](data=[Model(id='text-search-babbage-doc-001', created=1651172509, object='model', owned_by='openai-dev'),
                              Model(id='gpt-4', created=1687882411, object='model', owned_by='openai'), ...
        >>> client.model
        ['gpt3', 'gpt4']
        >>> client("Kubernetes에 대해 설명해", model="gpt3")
        'Kubernetes는 컨테이너화된 애플리케이션을 자동화하고 관리하기 위한 오픈소스 플랫폼입니다. ...
        >>> client("Kubernetes에 대해 설명해", stream=True)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284217669-043e3445-2e9a-4667-86af-e11f142ec931.gif
            :alt: client("Kubernetes에 대해 설명해", stream=True)
            :align: center
            :width: 600px
    """

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key)
        self.model_dict = {"gpt3": "gpt-3.5-turbo", "gpt4": "gpt-4"}
        self.model = list(self.model_dict.keys())

    def __call__(
        self,
        message: str,
        model: Optional[str] = "gpt4",
        stream: Optional[bool] = False,
    ):
        if stream:
            stream = self.chat.completions.create(
                model=self.model_dict[model],
                messages=[{"role": "user", "content": message}],
                stream=stream,
            )
            cts = ""
            for part in stream:
                content = part.choices[0].delta.content or ""
                cts += content
                print(content, end="", flush=True)
            print()
        else:
            chat_completion = self.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model=self.model_dict[model],
            )
            cts = chat_completion.choices[0].message.content
        return cts
