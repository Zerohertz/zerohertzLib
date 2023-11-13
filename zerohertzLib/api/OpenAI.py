import openai


def api_key(api_key: str) -> openai.openai_object.OpenAIObject:
    """OpenAI의 API 키 등록

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280564436-27d62c06-f59a-4772-8cd6-48be49147a86.png
        :alt: OpenAI API Key
        :align: center

    위와 같이 `OpenAI <https://platform.openai.com/account/api-keys>`_ 페이지에서 발급 후 API를 등록해야 사용할 수 있다.

    Args:
        api_key (``str``): OpenAI의 API Key

    Returns:
        ``openai.openai_object.OpenAIObject``: 사용 가능한 model들의 정보

    Examples:
        >>> zz.api.api_key("sk-...")
        <OpenAIObject list at ...> JSON: {
            "object": "list",
            "data": [
                {
                "id": "text-search-babbage-doc-001",
                "object": "model",
                "created": ...,
                "owned_by": "openai-dev"
                },
                {
                "id": "gpt-3.5-turbo-16k-0613",
                "object": "model",
                "created": ...,
                "owned_by": "openai"
                },
                ...
    """
    openai.api_key = api_key
    return openai.Model.list()


def gpt(message: str) -> str:
    """GPT 3.5 실행

    Args:
        message (``str``): ChatGPT 3.5의 입력

    Returns:
        ``str``: ChatGPT 3.5의 출력

    Examples:
        >>> zz.api.api_key("sk-...")
        >>> zz.api.gpt("hi")
        'Hello! How can I assist you today?'
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
    )
    return completion.choices[0].message.content
