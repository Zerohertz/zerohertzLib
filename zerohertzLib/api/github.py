# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os
import re
from collections import defaultdict
from typing import Any, Literal

import requests

from zerohertzLib.util import rmtree


class GitHub:
    """GitHub API를 사용하기 위한 class

    Args:
        user: GitHub API를 호출할 user
        repo: GitHub API를 호출할 repository
        token: GitHub의 token
        issue: `True`: Issue & PR, `False`: Only PR

    Examples:
        >>> gh = zz.api.GitHub("Zerohertz", "zerohertzLib", token="ghp_...")
        >>> fix = gh("fix", 20)
        >>> len(fix)
        20
        >>> fix[0].keys()
        dict_keys(['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'body', 'reactions', 'timeline_url', 'performed_via_github_app', 'state_reason'])
    """

    def __init__(
        self,
        user: str = "Zerohertz",
        repo: str = "zerohertzLib",
        token: str | None = None,
        issue: bool = True,
    ) -> None:
        if token is None:
            self.headers = {
                "Accept": "application/vnd.github.v3+json",
            }
        else:
            self.headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            }
        self.user = user
        self.repo = repo
        self.issue = issue
        if issue:
            self.url = f"https://api.github.com/repos/{user}/{repo}/issues"
        else:
            self.url = f"https://api.github.com/repos/{user}/{repo}/pulls"

    def __call__(
        self,
        lab: str = "all",
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """
        API 호출 수행

        Args:
            lab: 선택할 GitHub repository의 label (`issue=False` 시 error 발생)
            per_page: 1회 호출 시 출력될 결과의 수

        Returns:
            API 호출 결과
        """
        results = []
        page = 1
        total_fetched = 0
        while True:
            params = {
                "state": "all",
                "sort": "created",
                "direction": "desc",
                "per_page": per_page,
                "page": page,
            }
            if lab != "all":
                if not self.issue:
                    raise ValueError(
                        "If you want to filter by label, use\n\t--->\tGitHub(issue=True)\t<---"
                    )
                params["labels"] = lab
            response = requests.get(
                self.url, headers=self.headers, params=params, timeout=10
            )
            if response.status_code != 200:
                raise OSError(
                    f"GitHub API Response: {response.status_code}\n\t{response.json()}"
                )
            data = response.json()
            results.extend(data)
            total_fetched += len(data)
            if len(data) < per_page:
                break
            page += 1
        # ISSUE
        # dict_keys(['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'body', 'reactions', 'timeline_url', 'performed_via_github_app', 'state_reason'])
        # PULL REQUEST
        # dict_keys(['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'draft', 'pull_request', 'body', 'reactions', 'timeline_url', 'performed_via_github_app', 'state_reason'])

        # html_url: HTML의 URL                         https://github.com/Zerohertz/zerohertzLib/pull/64
        # number: Issue 또는 PR의 번호                 64
        # title: Issue 또는 PR의 제목                  [Docs] Build by Sphinx for GitHub Pages
        # body: Issue 또는 PR의 MarkDown               #63 (Build: 6095f8f85a0d6d8936a2caa373e675c6f5368644)
        # labels: Issue 또는 PR에 할당된 label들       list[dict_keys(['id', 'node_id', 'url', 'name', 'color', 'default', 'description'])]
        # closed_at: Issue 또는 PR이 종료된 시점       2023-11-16T07:48:51Z
        return results

    def _shield_icon(self, tag: str, color: str, href: str) -> str:
        return f"""<a href="{href}"><img src="https://img.shields.io/badge/{tag}-{color}?style=flat-square&logo=github" alt="{tag}"/></a>\n"""

    def _labels_markdown(self, labels: list[dict[str, Any]]) -> str:
        labels_markdown = """<p align="center">\n"""
        for label in labels:
            tag, color, href = (
                label["name"],
                label["color"],
                f"https://github.com/Zerohertz/zerohertzLib/pulls?q=is:pr label:{label['name']}",
            )
            labels_markdown += self._shield_icon(tag, color, href)
        labels_markdown += "</p>\n"
        return labels_markdown

    def _replace_cancel_line(self, body: str) -> str:
        return re.sub(r"~(.*?)~", r"<s>\1</s>", body)

    def _adjust_mkdocs_indent(self, body: str) -> str:
        """MkDocs format에 맞게 indent를 4칸 단위로 조정"""
        lines = body.split("\n")
        adjusted_lines = []
        for line in lines:
            stripped = line.lstrip()
            if line == stripped:  # indent가 없는 경우
                adjusted_lines.append(line)
                continue
            current_indent = len(line) - len(stripped)
            if current_indent % 4 == 0:
                adjusted_lines.append(line)
            else:
                target_level = (current_indent + 3) // 4  # 올림 처리
                new_indent = "    " * target_level
                adjusted_lines.append(new_indent + stripped)
        return "\n".join(adjusted_lines)

    def _replace_issue(self, body: str) -> str:
        return re.sub(
            r"#(\d+)",
            lambda match: f"""<a href="https://github.com/{self.user}/{self.repo}/issues/{match.group(1)}">#{match.group(1)}</a>""",
            body,
        )

    def _replace_pr_title(self, body: str) -> str:
        return re.sub(r"### (.*?)\n", r"<h4>\1</h4>\n", body)

    def _merge_release_note_version(
        self, version: str, data: list[list[Any]], tool: Literal["sphinx", "mkdocs"]
    ) -> str:
        merge_release_note = f"## {version}\n\n"
        for idx, (
            number,
            html_url,
            labels,
            title,
            updated_at,
            closed_at,
            body,
        ) in enumerate(data):
            merge_release_note += (
                f"""<h3>{title} (<a href={html_url}>#{number}</a>)</h3>\n\n"""
            )
            if closed_at is None:
                date = updated_at.split("T")[0].replace("-", "/")
            else:
                date = closed_at.split("T")[0].replace("-", "/")
            if tool == "sphinx":
                merge_release_note += "`{admonition} Release Date\n"
                merge_release_note += f":class: tip\n\n{date}\n```\n\n"
            else:
                merge_release_note += '!!! tip "Release Date"\n'
                merge_release_note += f"    {date}\n\n"
            merge_release_note += f"{self._labels_markdown(labels)}\n\n"
            if body is not None:
                body = body.replace("\r\n", "\n")
                body = self._replace_cancel_line(body)
                if tool == "mkdocs":
                    body = self._adjust_mkdocs_indent(body)
                body = self._replace_issue(body)
                body = self._replace_pr_title(body)
                merge_release_note += body
                if idx < len(data) - 1:
                    merge_release_note += "\n\n"
        return merge_release_note

    def _parse_version(self, title: str) -> str:
        version = re.findall(r"\[(.*?)\]", title)
        assert len(version) == 1
        return version[0]

    def _write_release_note_version(
        self, name: str, path: str, version: str, body: str
    ) -> None:
        with open(f"{path}/{name}/{version}.md", "w", encoding="utf-8") as file:
            file.writelines(f"# {version}\n\n{body}")

    def _write_release_note(self, name: str, path: str, versions: list[str]) -> None:
        release_note_body = (
            "# Release Notes\n\n```{eval-rst}\n.. toctree::\n\t:maxdepth: 1\n\n"
        )
        for version in versions:
            release_note_body += f"\t{name}/{version}\n"
        release_note_body += "```\n"
        with open(f"{path}/{name}.md", "w", encoding="utf-8") as file:
            file.writelines(release_note_body)

    def release_note(
        self,
        name: str = "release",
        path: str = "docs",
        tool: Literal["sphinx", "mkdocs"] = "mkdocs",
    ) -> None:
        """
        Args:
            name: Release note file 및 directory의 이름
            path: Release note를 작성할 경로
            tool: Release note를 배포할 tool

        Examples:
            >>> gh = zz.api.GitHub("Zerohertz", "zerohertzLib", token="ghp_...")
            >>> gh.release_note()
        """
        releases = self("release") + self("release/chore")
        bodies_version = defaultdict(list)
        versions = defaultdict(str)
        for release in releases:
            if "\x08" in release["title"]:
                error_title = release["title"].replace("\x08", "-> \\x08 <-")
                raise ValueError(
                    f"""\\x08 is in '{release["html_url"]}'\n\tError Title: {error_title}"""
                )
            if (
                release["state"] == "closed"
                and release["pull_request"]["merged_at"] is None
            ):
                continue
            version = self._parse_version(release["title"])
            bodies_version[version].append(
                [
                    release["number"],
                    release["html_url"],
                    release["labels"],
                    release["title"],
                    release["updated_at"],
                    release["closed_at"],
                    release["body"],
                ]
            )
        for data in bodies_version.values():
            data.sort(reverse=True)
        rmtree(os.path.join(path, name))
        for version, data in bodies_version.items():
            ver = ".".join(version.split(".")[:-1])
            body = self._merge_release_note_version(version, data, tool)
            if ver in versions:
                versions[ver] += "\n\n" + body
            else:
                versions[ver] = body
        for version, body in versions.items():
            self._write_release_note_version(name, path, version, body)
        if tool == "sphinx":
            self._write_release_note(name, path, list(versions.keys()))
