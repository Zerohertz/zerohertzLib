void setBuildStatus(String message, String state, String context) {
    step([
        $class: "GitHubCommitStatusSetter",
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/Zerohertz/zerohertzLib"],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: context],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}

pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    jenkins/agent-type: python
spec:
  containers:
    - name: jnlp
      image: jenkins/inbound-agent:latest
      resources:
        requests:
          memory: "512Mi"
          cpu: "500m"
        limits:
          memory: "1024Mi"
          cpu: "1000m"
    - name: python
      image: python:3.11
      command:
        - cat
      tty: true
      resources:
        requests:
          memory: "2048Mi"
          cpu: "2000m"
        limits:
          memory: "4096Mi"
          cpu: "4000m"
            """
        }
    }

    stages {
        stage("Setup") {
            steps {
                script {
                    def status = sh(script: "git log -1 --pretty=%B ${env.GIT_COMMIT}", returnStdout: true, returnStatus: true)
                    if (status != 0) {
                        commitMessage = ""
                        currentBuild.result = "SUCCESS"
                    } else {
                        commitMessage = sh(script: "git log -1 --pretty=%B ${env.GIT_COMMIT}", returnStdout: true).trim()
                    }
                    slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                }
            }
        }
        // [`*` Push] "Merge pull request*/docs"
        // [`*` Push] "Merge pull request*[Docs] Build by Sphinx for GitHub Pages"
        stage("Merge From Docs") {
            when {
                expression {
                    return commitMessage.startsWith("Merge pull request") && (commitMessage.endsWith("Zerohertz/docs") || commitMessage.endsWith("[Docs] Build by Sphinx for GitHub Pages"))
                }
            }
            steps {
                script {
                    setBuildStatus("Success", "SUCCESS", "1. Lint")
                    setBuildStatus("Success", "SUCCESS", "2. Build")
                    setBuildStatus("Success", "SUCCESS", "3. Test")
                    setBuildStatus("Success", "SUCCESS", "4. Docs")
                    setBuildStatus("Success", "SUCCESS", "$STAGE_NAME")
                    slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                }
            }
        }
        // [`dev*` Push]
        // [`master` PR] (Except "Merge pull request*/docs" && "Merge pull request*[Docs] Build by Sphinx for GitHub Pages")
        stage("1. Lint") {
            when {
                anyOf {
                    branch pattern: "dev.*", comparator: "REGEXP"
                    expression {
                        def isMasterPR = env.CHANGE_TARGET == "master"
                        def isNotDocsMerge = !(commitMessage.startsWith("Merge pull request") && (commitMessage.endsWith("Zerohertz/docs") || commitMessage.endsWith("[Docs] Build by Sphinx for GitHub Pages")))
                        return isMasterPR && isNotDocsMerge
                    }
                }
            }
            steps {
                script {
                    try {
                        def startTime = System.currentTimeMillis()
                        setBuildStatus("Checking Lint...", "PENDING", "$STAGE_NAME")
                        container("python") {
                            sh "pip install .'[all]'"
                            sh "pip install -r requirements/requirements-style.txt"
                            sh "black --check ."
                            sh "flake8 zerohertzLib"
                            sh "pylint -r n zerohertzLib"
                        }
                        def endTime = System.currentTimeMillis()
                        def DURATION = (endTime - startTime) / 1000
                        setBuildStatus("Successful in ${DURATION}s", "SUCCESS", "$STAGE_NAME")
                        slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                    } catch (Exception e) {
                        def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                        setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                        slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}\nError Message: ${STAGE_ERROR_MESSAGE}")
                        throw e
                    }
                }
            }
        }
        // [`master` Push] (Except "Merge pull request*/chore*")
        // [`dev*` Push]
        // [`master` PR] (Except "Merge pull request*/docs" && "Merge pull request*[Docs] Build by Sphinx for GitHub Pages")
        stage("2. Build") {
            when {
                anyOf {
                    expression {
                        def isMasterBranch = env.BRANCH_NAME == "master"
                        def isNotChoreMerge = !(commitMessage.startsWith("Merge pull request") && commitMessage.contains("/chore"))
                        return isMasterBranch && isNotChoreMerge
                    }
                    branch pattern: "dev.*", comparator: "REGEXP"
                    expression {
                        def isMasterPR = env.CHANGE_TARGET == "master"
                        def isNotDocsMerge = !(commitMessage.startsWith("Merge pull request") && (commitMessage.endsWith("Zerohertz/docs") || commitMessage.endsWith("[Docs] Build by Sphinx for GitHub Pages")))
                        return isMasterPR && isNotDocsMerge
                    }
                }
            }
            steps {
                script {
                    try {
                        def startTime = System.currentTimeMillis()
                        setBuildStatus("Build...", "PENDING", "$STAGE_NAME")
                        container("python") {
                            sh "apt update"
                            sh "pip install build"
                            sh "python -m build ."
                        }
                        def endTime = System.currentTimeMillis()
                        def DURATION = (endTime - startTime) / 1000
                        setBuildStatus("Successful in ${DURATION}s", "SUCCESS", "$STAGE_NAME")
                        slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                    } catch (Exception e) {
                        def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                        setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                        slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}\nError Message: ${STAGE_ERROR_MESSAGE}")
                        throw e
                    }
                }
            }
        }
        // [`dev*` Push]
        // [`master` PR] (Except "Merge pull request*/docs" && "Merge pull request*[Docs] Build by Sphinx for GitHub Pages")
        stage("3. Test") {
            when {
                anyOf {
                    branch pattern: "dev.*", comparator: "REGEXP"
                    expression {
                        def isMasterPR = env.CHANGE_TARGET == "master"
                        def isNotDocsMerge = !(commitMessage.startsWith("Merge pull request") && (commitMessage.endsWith("Zerohertz/docs") || commitMessage.endsWith("[Docs] Build by Sphinx for GitHub Pages")))
                        return isMasterPR && isNotDocsMerge
                    }
                }
            }
            steps {
                withCredentials([string(credentialsId: "OpenAI_Token", variable: "OPENAI_TOKEN"),
                                string(credentialsId: "Discord_Webhook_URL", variable: "DISCORD_WEBHOOK_URL"),
                                string(credentialsId: "Slack_Webhook_URL", variable: "SLACK_WEBHOOK_URL"),
                                string(credentialsId: "Slack_Bot_Token", variable: "SLACK_BOT_TOKEN")]) {
                    script {
                        try {
                            def startTime = System.currentTimeMillis()
                            setBuildStatus("Test...", "PENDING", "$STAGE_NAME")
                            container("python") {
                                sh "apt install python3-opencv -y"
                                sh "pip install pytest"
                                sh "pytest"
                            }
                            def endTime = System.currentTimeMillis()
                            def DURATION = (endTime - startTime) / 1000
                            setBuildStatus("Successful in ${DURATION}s", "SUCCESS", "$STAGE_NAME")
                            slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                        } catch (Exception e) {
                            def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                            setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                            slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}\nError Message: ${STAGE_ERROR_MESSAGE}")
                            throw e
                        }
                    }
                }
            }
        }
        // [`master` PR] (Except "Merge pull request*/docs" && "Merge pull request*[Docs] Build by Sphinx for GitHub Pages")
        stage("4. Docs") {
            when {
                expression {
                    def isMasterPR = env.CHANGE_TARGET == "master"
                    def isNotDocsMerge = !(commitMessage.startsWith("Merge pull request") && (commitMessage.endsWith("Zerohertz/docs") || commitMessage.endsWith("[Docs] Build by Sphinx for GitHub Pages")))
                    return isMasterPR && isNotDocsMerge
                }
            }
            steps {
                script {
                    try {
                        def startTime = System.currentTimeMillis()
                        setBuildStatus("Build...", "PENDING", "$STAGE_NAME")
                        if (env.CHANGE_BRANCH.startsWith("dev-")) {
                            sh "sed -i 's/^__version__ = .*/__version__ = \"'${env.CHANGE_BRANCH.replace('dev-', '')}'\"/' zerohertzLib/__init__.py"
                        } else if (env.CHANGE_BRANCH.startsWith("chore-")) {
                            echo "No action required for chore- branch"
                        } else {
                            error "Unsupported branch type: ${env.CHANGE_BRANCH}"
                        }
                        withCredentials([usernamePassword(credentialsId: "GitHub", usernameVariable: "GIT_USERNAME", passwordVariable: "GIT_PASSWORD")]) {
                            sh '''
                            git config --global user.email "ohg3417@gmail.com"
                            git config --global user.name "${GIT_USERNAME}"
                            git config --global credential.helper "!f() { echo username=${GIT_USERNAME}; echo password=${GIT_PASSWORD}; }; f"
                            '''
                            def isExistDocsRemote = sh(returnStdout: true, script: "git ls-remote --heads origin docs").trim()
                            if (isExistDocsRemote) {
                                sh "git push origin --delete docs"
                            }
                            sh "git checkout -b docs"
                            def hasVersionChanges = sh(
                                script: "git diff --name-only | grep -w zerohertzLib/__init__.py",
                                returnStatus: true
                            ) == 0
                            if (hasVersionChanges) {
                                sh "git add zerohertzLib/__init__.py"
                                sh "git commit -m ':hammer: Update: Version (#${env.CHANGE_ID})'"
                                sh "git push origin docs"
                            }
                            container("python") {
                                sh "apt update"
                                sh "apt install build-essential -y"
                                sh "pip install -r requirements/requirements-docs.txt"
                                sh 'python sphinx/release_note.py --token $GIT_PASSWORD'
                                sh "python sphinx/example_images.py"
                                sh "cd sphinx && make html"
                                sh "rm -rf docs"
                                sh "mv sphinx/build/html docs"
                                sh "touch docs/.nojekyll"
                            }
                            def hasDocsChanges = sh(
                                script: "git diff --name-only | grep -w docs",
                                returnStatus: true
                            ) == 0
                            def hasSphinxChanges = sh(
                                script: "git diff --name-only | grep -w sphinx/source",
                                returnStatus: true
                            ) == 0
                            if (hasDocsChanges || hasSphinxChanges) {
                                sh "git add docs"
                                sh "git add sphinx/source"
                                sh "git commit -m ':memo: Docs: Build Sphinx (#${env.CHANGE_ID})'"
                                sh "git push origin docs"
                                sh """
                                echo '{
                                    "title": "[Docs] Build by Sphinx for GitHub Pages",
                                    "head": "docs",
                                    "base": "${env.CHANGE_BRANCH}",
                                    "body": "#${env.CHANGE_ID} (Build: ${env.GIT_COMMIT})"
                                }' > payload.json
                                """
                                sh '''
                                curl -X POST -H "Authorization: token $GIT_PASSWORD" \
                                -H "Accept: application/vnd.github.v3+json" \
                                https://api.github.com/repos/Zerohertz/zerohertzLib/pulls \
                                -d @payload.json
                                '''
                            } else {
                                setBuildStatus("Success", "SUCCESS", "Merge From Docs")
                            }
                        }
                        def endTime = System.currentTimeMillis()
                        def DURATION = (endTime - startTime) / 1000
                        setBuildStatus("Successful in ${DURATION}s", "SUCCESS", "$STAGE_NAME")
                        slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                    } catch (Exception e) {
                        def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                        setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                        slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}\nError Message: ${STAGE_ERROR_MESSAGE}")
                        throw e
                    }
                }
            }
        }
        // [`master` Push] "Merge pull request*from Zerohertz/dev*" (Except "*from Zerohertz/chore*")
        stage("Deploy") {
            when {
                expression {
                    def isMasterBranch = env.BRANCH_NAME == "master"
                    def isPR = commitMessage.startsWith("Merge pull request")
                    def isDevMerge = commitMessage.contains("from Zerohertz/dev")
                    def isChoreMerge = commitMessage.contains("from Zerohertz/chore")
                    return isMasterBranch && isPR && isDevMerge && !isChoreMerge
                }
            }
            steps {
                script {
                    parallel(
                        PyPI: {
                            stage("PyPI") {
                                try {
                                    def startTime = System.currentTimeMillis()
                                    setBuildStatus("Deploy...", "PENDING", "$STAGE_NAME")
                                    container("python") {
                                        sh "pip install twine"
                                        withCredentials([usernamePassword(credentialsId: "PyPI", usernameVariable: "PYPI_USERNAME", passwordVariable: "PYPI_PASSWORD")]) {
                                            sh 'twine upload -u $PYPI_USERNAME -p $PYPI_PASSWORD dist/*'
                                        }
                                    }
                                    def endTime = System.currentTimeMillis()
                                    def DURATION = (endTime - startTime) / 1000
                                    setBuildStatus("Successful in ${DURATION}s", "SUCCESS", "$STAGE_NAME")
                                    slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                                } catch (Exception e) {
                                    def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                                    setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                                    slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}\nError Message: ${STAGE_ERROR_MESSAGE}")
                                    throw e
                                }
                            }
                        },
                        GitHub: {
                            stage("GitHub") {
                                try {
                                    def startTime = System.currentTimeMillis()
                                    setBuildStatus("Deploy...", "PENDING", "$STAGE_NAME")
                                    def PACKAGE_VERSION = ""
                                    container("python") {
                                        sh "apt update"
                                        sh "apt install python3-opencv -y"
                                        sh "pip install ."
                                        PACKAGE_VERSION = sh(
                                            script: 'python -c "import zerohertzLib; print(zerohertzLib.__version__)"',
                                            returnStdout: true
                                        ).trim()
                                    }
                                    withCredentials([usernamePassword(credentialsId: "GitHub", usernameVariable: "GIT_USERNAME", passwordVariable: "GIT_PASSWORD")]) {
                                        sh '''
                                        git config --global user.email "ohg3417@gmail.com"
                                        git config --global user.name "${GIT_USERNAME}"
                                        git config --global credential.helper "!f() { echo username=${GIT_USERNAME}; echo password=${GIT_PASSWORD}; }; f"
                                        '''
                                        sh "git tag ${PACKAGE_VERSION}"
                                        sh "git push origin ${PACKAGE_VERSION}"
                                    }
                                    def endTime = System.currentTimeMillis()
                                    def DURATION = (endTime - startTime) / 1000
                                    setBuildStatus("Successful in ${DURATION}s", "SUCCESS", "$STAGE_NAME")
                                    slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
                                } catch (Exception e) {
                                    def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                                    setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                                    slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}\nError Message: ${STAGE_ERROR_MESSAGE}")
                                    throw e
                                }
                            }
                        }
                    )
                }
            }
        }
    }
}
