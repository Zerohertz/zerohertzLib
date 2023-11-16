# CI/CD Pipelines

<p align="center">
    <img width="1844" alt="CI/CD" src="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/283340051-f8e81344-e662-4cac-9e8f-fb722997dbff.png">
</p>

|Stage|AS-IS|TO-BE|
|:-:|-|-|
|1. `Setup`|⭕ [`*` Push]|⭕ [`*` Push]|
|2. `Merge From Docs`|👎 [`*` Push] "Merge pull request\*/docs\*"|👍 [`*` Push] "Merge pull request\*/docs"|
|3. `1. Lint`|⭕ [`dev*` Push]<br>👎 [`master` PR] (Except "Merge pull request\*/docs\*")|⭕ [`dev*` Push]<br>👍 [`master` PR] (Except "Merge pull request\*/docs")|
|4. `2. Build`|👎 [`master` Push]</br>⭕ [`dev*` Push]<br>👎 [`master` PR] (Except "Merge pull request\*/docs\*")|👍 [`master` Push] (Except "Merge pull request\*/docs\*")</br>⭕ [`dev*` Push]<br>👍 [`master` PR] (Except "Merge pull request\*/docs")|
|5. `3. Test`|⭕ [`dev*` Push]<br>👎 [`master` PR] (Except "Merge pull request\*/docs\*")|⭕ [`dev*` Push]<br>👍 [`master` PR] (Except "Merge pull request\*/docs")|
|6. `4. Docs`|👎 [`master` PR] (Except "Merge pull request\*/docs\*")|👍 [`master` PR] (Except "Merge pull request\*/docs")|
|7. `Deploy`|👎 [`master` Push]|👍 [`master` Push] (Except "Merge pull request\*/docs\*")|
