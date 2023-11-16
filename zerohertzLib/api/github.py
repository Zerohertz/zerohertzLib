import os
import re
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import requests


class GitHub:
    """GitHub API를 사용하기 위한 클래스

    Args:
        user (``Optional[str]``): GitHub API를 호출할 user
        repo (``Optional[str]``): GitHub API를 호출할 repository
        token (``Optional[str]``): GitHub의 token
        issue (``Optional[bool]``): ``True``: Issue & PR, ``False``: Only PR

    Methods:
        __call__:
            API 호출 수행

            Args:
                lab (``Optional[str]``): 선택할 GitHub repository의 label (``issue=False`` 시 에러 발생)
                per_page (``Optional[int]``): 1회 호출 시 출력될 결과의 수

            Returns:
                ``List[Dict[str, Any]]``: API 호출 결과

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
        user: Optional[str] = "Zerohertz",
        repo: Optional[str] = "zerohertzLib",
        token: Optional[str] = None,
        issue: Optional[bool] = True,
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
        self, lab: Optional[str] = "all", per_page: Optional[int] = 100
    ) -> List[Dict[str, Any]]:
        if lab == "all":
            params = {
                "state": "all",
                "sort": "created",
                "direction": "desc",
                "per_page": per_page,
            }
        else:
            if not self.issue:
                raise Exception(
                    "If you want to filter by label, use\n\t--->\tGitHub(issue=True)\t<---"
                )
            params = {
                "state": "all",
                "sort": "created",
                "direction": "desc",
                "per_page": per_page,
                "labels": lab,
            }
        response = requests.get(self.url, headers=self.headers, params=params)
        if not response.status_code == 200:
            raise Exception(
                f"GitHub API Response: {response.status_code}\n\t{response.json()}"
            )
        results = response.json()
        """
        ISSUE
        dict_keys(['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'body', 'reactions', 'timeline_url', 'performed_via_github_app', 'state_reason'])
        PULL REQUEST
        dict_keys(['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'draft', 'pull_request', 'body', 'reactions', 'timeline_url', 'performed_via_github_app', 'state_reason'])
        
        html_url: HTML의 URL                         https://github.com/Zerohertz/zerohertzLib/pull/64
        number: Issue 또는 PR의 번호                   64
        title: Issue 또는 PR의 제목                    [Docs] Build by Sphinx for GitHub Pages
        body: Issue 또는 PR의 MarkDown                #63 (Build: 6095f8f85a0d6d8936a2caa373e675c6f5368644)
        labels: Issue 또는 PR에 할당된 label들          List[dict_keys(['id', 'node_id', 'url', 'name', 'color', 'default', 'description'])]
        closed_at: Issue 또는 PR이 종료된 시점           2023-11-16T07:48:51Z
        """
        return results

    def _parse_version(self, title):
        version = re.findall(r"\[(.*?)\]", title)
        assert len(version) == 1
        return version[0]

    def _parse_issues_from_pr_body(self, body: str) -> Set[str]:
        issues = re.findall(r"#(\d+)", body)
        return set(issues)

    def _replace_issue_markdown(self, body: str, issue: str) -> str:
        return body.replace(
            f"#{issue}",
            f"[#{issue}](https://github.com/{self.user}/{self.repo}/issues/{issue})",
        )

    def _shield_icon(self, tag: str, color: str, href: str):
        return f"""<a href="{href}"><img src="https://img.shields.io/badge/{tag}-{color}?style=flat-square&logo=github" alt="{tag}"/></a>\n"""

    def _labels_markdown(self, labels: List[Dict[str, Any]]) -> str:
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

    def _merge_release_note_version(self, version: str, data: List[List[Any]]) -> str:
        merge_release_note = f"# {version}\n\n"
        for number, html_url, labels, title, updated_at, closed_at, body in data:
            merge_release_note += f"## {title} ([#{number}]({html_url}))\n\n"
            if closed_at is None:
                date = updated_at.split("T")[0].replace("-", "/")
            else:
                date = closed_at.split("T")[0].replace("-", "/")
            merge_release_note += "```{admonition} Release Date\n"
            merge_release_note += f":class: tip\n\n{date}\n```\n\n"
            merge_release_note += f"{self._labels_markdown(labels)}\n\n"
            if body is not None:
                issues = self._parse_issues_from_pr_body(body)
                for issue in issues:
                    body = self._replace_issue_markdown(body, issue)
                merge_release_note += body
        return merge_release_note

    def _write_release_note_version(
        self, name: str, sphinx_source_path: str, version: str, body: str
    ) -> None:
        with open(
            f"{sphinx_source_path}/{name}/{version}.md", "w", encoding="utf-8"
        ) as f:
            f.writelines(body)

    def _write_release_note(
        self, name: str, sphinx_source_path: str, versions: List[str]
    ) -> None:
        release_note_body = "# Release Notes\n\n```{eval-rst}\n.. toctree::\n\t:maxdepth: 2\n\t:caption: Contents:\n\n"
        for version in versions:
            release_note_body += f"\t{name}/{version}\n"
        release_note_body += "```\n"
        with open(f"{sphinx_source_path}/{name}.md", "w", encoding="utf-8") as f:
            f.writelines(release_note_body)

    def release_note(
        self,
        name: Optional[str] = "release",
        sphinx_source_path: Optional[str] = "sphinx/source",
    ) -> None:
        """
        Args:
            name (``Optional[str]``): Release note file 및 directory의 이름
            sphinx_source_path (``Optional[str]``): Sphinx의 ``source`` 경로

        Returns:
            ``List[requests.models.Response]``: Discord Webhook의 응답

        Examples:
            >>> gh = zz.api.GitHub("Zerohertz", "zerohertzLib", token="ghp_...")
            >>> gh.release_note(sphinx_source_path=os.path.join(sphinx, "source"))
        """
        releases = self("release") + self("release/docs")
        bodies_version = defaultdict(list)
        for release in releases:
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
            data.sort()
        self._write_release_note(name, sphinx_source_path, list(bodies_version.keys()))
        try:
            shutil.rmtree(f"{sphinx_source_path}/{name}")
        except:
            pass
        os.mkdir(f"{sphinx_source_path}/{name}")
        for version, data in bodies_version.items():
            body = self._merge_release_note_version(version, data)
            self._write_release_note_version(name, sphinx_source_path, version, body)
