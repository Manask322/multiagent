import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import os
import requests
from typing import Optional, List, Dict, Any

# Set environment variables for Azure OpenAI
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://docs-search-aus-east.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "e2c8b179c3a24127a0ab3b38509fb9b1"

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (41 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


def github_search(query: str, search_type: str = "repositories", limit: int = 5) -> dict:
    """Search GitHub for repositories, users, or issues.
    
    Args:
        query (str): The search query.
        search_type (str): Type of search to perform. Options: "repositories", "users", "issues", "code".
        limit (int): Maximum number of results to return (default 5).
        
    Returns:
        dict: Status and search results or error message.
    """
    # GitHub API endpoints
    endpoints = {
        "repositories": "https://api.github.com/search/repositories",
        "users": "https://api.github.com/search/users",
        "issues": "https://api.github.com/search/issues",
        "code": "https://api.github.com/search/code"
    }
    
    # Validate search type
    if search_type not in endpoints:
        return {
            "status": "error",
            "error_message": f"Invalid search type: {search_type}. Valid options are: {', '.join(endpoints.keys())}"
        }
    
    # Set up GitHub API request
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Add auth token if available
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    # Make the API request
    try:
        response = requests.get(
            endpoints[search_type],
            headers=headers,
            params={"q": query, "per_page": limit}
        )
        response.raise_for_status()
        data = response.json()
        
        # Format the results based on search type
        results = []
        items = data.get("items", [])
        
        if search_type == "repositories":
            for repo in items:
                results.append({
                    "name": repo["full_name"],
                    "description": repo.get("description", "No description"),
                    "stars": repo["stargazers_count"],
                    "url": repo["html_url"]
                })
        elif search_type == "users":
            for user in items:
                results.append({
                    "username": user["login"],
                    "profile": user["html_url"],
                    "type": user["type"]
                })
        elif search_type == "issues":
            for issue in items:
                results.append({
                    "title": issue["title"],
                    "state": issue["state"],
                    "url": issue["html_url"],
                    "repository": issue["repository_url"].split("/repos/")[1]
                })
        elif search_type == "code":
            for code in items:
                results.append({
                    "repository": code["repository"]["full_name"],
                    "path": code["path"],
                    "url": code["html_url"]
                })
        
        return {
            "status": "success",
            "total_count": data["total_count"],
            "results": results
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error_message": f"GitHub API error: {str(e)}"
        }


def github_repo_info(owner: str, repo: str) -> dict:
    """Get information about a specific GitHub repository.
    
    Args:
        owner (str): The owner (user or organization) of the repository.
        repo (str): The name of the repository.
        
    Returns:
        dict: Status and repository information or error message.
    """
    print(f"[DEBUG] github_repo_info called with: owner='{owner}', repo='{repo}'")
    
    # Set up GitHub API request
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Add auth token if available
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        print(f"[DEBUG] Using GitHub token: {github_token[:4]}...{github_token[-4:] if len(github_token) > 8 else ''}")
        headers["Authorization"] = f"token {github_token}"
    else:
        print("[DEBUG] No GitHub token found. Using unauthenticated requests (rate limits apply)")
    
    print(f"[DEBUG] Headers: {headers}")
    
    # Make the API request
    try:
        # Get basic repo info
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        print(f"[DEBUG] Requesting repo info from: {repo_url}")
        
        response = requests.get(repo_url, headers=headers)
        status_code = response.status_code
        print(f"[DEBUG] Repo info response status: {status_code}")
        
        # Check for common status codes
        if status_code == 404:
            print(f"[DEBUG] Repository not found: {owner}/{repo}")
            return {
                "status": "error",
                "error_message": f"Repository not found: {owner}/{repo}"
            }
        elif status_code == 403:
            print(f"[DEBUG] Rate limit exceeded or access forbidden")
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
            rate_limit_reset = response.headers.get('X-RateLimit-Reset', 'unknown')
            print(f"[DEBUG] Rate limit remaining: {rate_limit_remaining}, Reset at: {rate_limit_reset}")
        
        # Continue with request
        response.raise_for_status()
        repo_data = response.json()
        print(f"[DEBUG] Successfully received repo data with {len(repo_data)} fields")
        
        # Get latest commits
        commits_url = f"{repo_url}/commits"
        print(f"[DEBUG] Requesting commits from: {commits_url}")
        
        commits_response = requests.get(commits_url, headers=headers, params={"per_page": 3})
        print(f"[DEBUG] Commits response status: {commits_response.status_code}")
        
        commits_response.raise_for_status()
        commits_data = commits_response.json()
        print(f"[DEBUG] Successfully received {len(commits_data)} commits")
        
        # Format commit info
        commits = []
        for i, commit in enumerate(commits_data):
            print(f"[DEBUG] Processing commit #{i+1}: {commit['sha'][:7]}")
            try:
                commit_info = {
                    "sha": commit["sha"][:7],
                    "message": commit["commit"]["message"].split("\n")[0],
                    "author": commit["commit"]["author"]["name"],
                    "date": commit["commit"]["author"]["date"]
                }
                commits.append(commit_info)
                print(f"[DEBUG] Added commit: {commit_info}")
            except KeyError as e:
                print(f"[DEBUG] Error extracting commit data: {e}. Commit structure: {commit.keys()}")
        
        # Build result dictionary
        print(f"[DEBUG] Building final result dictionary")
        result = {
            "name": repo_data.get("full_name", f"{owner}/{repo}") if repo_data else f"{owner}/{repo}",
            "description": repo_data.get("description", "No description") if repo_data else "No description",
            "stars": repo_data.get("stargazers_count", 0) if repo_data else 0,
            "forks": repo_data.get("forks_count", 0) if repo_data else 0,
            "open_issues": repo_data.get("open_issues_count", 0) if repo_data else 0,
            "language": repo_data.get("language", "Not specified") if repo_data else "Not specified",
            "license": repo_data.get("license", {}).get("name", "No license") if repo_data and repo_data.get("license") else "No license",
            "created_at": repo_data.get("created_at", "Unknown") if repo_data else "Unknown",
            "updated_at": repo_data.get("updated_at", "Unknown") if repo_data else "Unknown",
            "url": repo_data.get("html_url", f"https://github.com/{owner}/{repo}") if repo_data else f"https://github.com/{owner}/{repo}",
            "latest_commits": commits if commits else []
        }
        
        print(f"[DEBUG] Result dictionary built successfully with {len(result)} fields")
        
        return {
            "status": "success",
            "repo_info": result
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"GitHub API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

def github_compare_commits(owner: str, repo: str, base: str, head: str) -> dict:
    """Compare two commits in a GitHub repository and show changes.
    
    Args:
        owner (str): The owner (user or organization) of the repository.
        repo (str): The name of the repository.
        base (str): The base commit SHA, branch, or tag to compare from.
        head (str): The head commit SHA, branch, or tag to compare to.
        
    Returns:
        dict: Status and comparison information or error message.
    """
    print(f"[DEBUG] github_compare_commits called with: owner='{owner}', repo='{repo}', base='{base}', head='{head}'")
    
    # Set up GitHub API request
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Add auth token if available
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        print(f"[DEBUG] Using GitHub token: {github_token[:4]}...{github_token[-4:] if len(github_token) > 8 else ''}")
        headers["Authorization"] = f"token {github_token}"
    else:
        print("[DEBUG] No GitHub token found. Using unauthenticated requests (rate limits apply)")
    
    print(f"[DEBUG] Headers: {headers}")
    
    # Make the API request
    try:
        # Construct the compare URL
        compare_url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
        print(f"[DEBUG] Requesting commit comparison from: {compare_url}")
        
        response = requests.get(compare_url, headers=headers)
        status_code = response.status_code
        print(f"[DEBUG] Compare response status: {status_code}")
        
        # Check for common status codes
        if status_code == 404:
            print(f"[DEBUG] Repository or commits not found: {owner}/{repo}, {base}...{head}")
            return {
                "status": "error",
                "error_message": f"Repository or commits not found: {owner}/{repo}, {base}...{head}"
            }
        elif status_code == 403:
            print(f"[DEBUG] Rate limit exceeded or access forbidden")
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
            rate_limit_reset = response.headers.get('X-RateLimit-Reset', 'unknown')
            print(f"[DEBUG] Rate limit remaining: {rate_limit_remaining}, Reset at: {rate_limit_reset}")
        elif status_code == 422:
            print(f"[DEBUG] Invalid comparison or commits not found: {base}...{head}")
            return {
                "status": "error",
                "error_message": f"Invalid comparison or commits not found: {base}...{head}"
            }
        
        # Continue with request
        response.raise_for_status()
        compare_data = response.json()
        print(f"[DEBUG] Successfully received comparison data with {len(compare_data)} fields")
        
        # Process commits
        commits = []
        for i, commit in enumerate(compare_data.get('commits', [])):
            print(f"[DEBUG] Processing commit #{i+1}: {commit['sha'][:7]}")
            try:
                commit_info = {
                    "sha": commit["sha"][:7],
                    "message": commit["commit"]["message"].split("\n")[0],
                    "author": commit["commit"]["author"]["name"],
                    "date": commit["commit"]["author"]["date"]
                }
                commits.append(commit_info)
                print(f"[DEBUG] Added commit: {commit_info}")
            except KeyError as e:
                print(f"[DEBUG] Error extracting commit data: {e}. Commit structure: {commit.keys()}")
        
        # Process file changes
        files = []
        for i, file in enumerate(compare_data.get('files', [])):
            print(f"[DEBUG] Processing file #{i+1}: {file.get('filename')}")
            try:
                file_info = {
                    "filename": file.get("filename"),
                    "status": file.get("status"),  # added, modified, removed
                    "additions": file.get("additions", 0),
                    "deletions": file.get("deletions", 0),
                    "changes": file.get("changes", 0)
                }
                
                # Add patch information if available (showing actual code changes)
                if "patch" in file:
                    patch_preview = file["patch"][:200] + "..." if len(file["patch"]) > 200 else file["patch"]
                    file_info["patch_preview"] = patch_preview
                    print(f"[DEBUG] Added patch preview of length {len(patch_preview)}")
                
                files.append(file_info)
                print(f"[DEBUG] Added file: {file_info['filename']} ({file_info['status']})")
            except KeyError as e:
                print(f"[DEBUG] Error extracting file data: {e}. File structure: {file.keys()}")
        
        # Build result dictionary
        print(f"[DEBUG] Building final result dictionary")
        result = {
            "repository": f"{owner}/{repo}",
            "base_commit": base,
            "head_commit": head,
            "url": compare_data.get("html_url"),
            "status": compare_data.get("status"),  # ahead, behind, identical, or diverged
            "ahead_by": compare_data.get("ahead_by", 0),
            "behind_by": compare_data.get("behind_by", 0),
            "total_commits": compare_data.get("total_commits", 0),
            "commits": commits,
            "total_changes": len(files),
            "files_changed": files
        }
        
        # Add summary stats
        additions = sum(file.get("additions", 0) for file in compare_data.get('files', []))
        deletions = sum(file.get("deletions", 0) for file in compare_data.get('files', []))
        result["additions"] = additions
        result["deletions"] = deletions
        result["total_line_changes"] = additions + deletions
        
        print(f"[DEBUG] Result dictionary built successfully with {len(result)} fields")
        print(f"[DEBUG] Summary: {result['total_commits']} commits, {len(files)} files changed, +{additions}/-{deletions}")
        
        return {
            "status": "success",
            "comparison": result
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"GitHub API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

def github_explain_changes(owner: str, repo: str, base: str, head: str) -> dict:
    """Get and explain code changes between two commits in a GitHub repository.
    
    Args:
        owner (str): The owner (user or organization) of the repository.
        repo (str): The name of the repository.
        base (str): The base commit SHA, branch, or tag to compare from.
        head (str): The head commit SHA, branch, or tag to compare to.
        
    Returns:
        dict: Status and explanation of changes or error message.
    """
    print(f"[DEBUG] github_explain_changes called with: owner='{owner}', repo='{repo}', base='{base}', head='{head}'")
    
    # Set up GitHub API request
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Add auth token if available
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        print(f"[DEBUG] Using GitHub token: {github_token[:4]}...{github_token[-4:] if len(github_token) > 8 else ''}")
        headers["Authorization"] = f"token {github_token}"
    else:
        print("[DEBUG] No GitHub token found. Using unauthenticated requests (rate limits apply)")
    
    # Make the API request
    try:
        # Fetch the comparison data
        compare_url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
        print(f"[DEBUG] Requesting commit comparison from: {compare_url}")
        
        response = requests.get(compare_url, headers=headers)
        status_code = response.status_code
        print(f"[DEBUG] Compare response status: {status_code}")
        
        if status_code != 200:
            print(f"[DEBUG] Error response: {response.text}")
            return {
                "status": "error",
                "error_message": f"GitHub API error: Status {status_code} - {response.text[:200]}"
            }
        
        compare_data = response.json()
        print(f"[DEBUG] Successfully received comparison data")
        
        # Extract commits
        commits = []
        for commit in compare_data.get('commits', []):
            commits.append({
                "sha": commit["sha"][:7],
                "message": commit["commit"]["message"].split("\n")[0],
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"]
            })
        
        # Process file changes with full patches
        file_changes = []
        total_additions = 0
        total_deletions = 0
        
        for file in compare_data.get('files', []):
            change = {
                "filename": file.get("filename"),
                "status": file.get("status"),  # added, modified, removed
                "additions": file.get("additions", 0),
                "deletions": file.get("deletions", 0),
                "changes": file.get("changes", 0)
            }
            
            # Add full patch information (actual code changes)
            if "patch" in file:
                change["patch"] = file["patch"]
            
            file_changes.append(change)
            total_additions += file.get("additions", 0)
            total_deletions += file.get("deletions", 0)
        
        # Create a summary of changes
        summary = {
            "repository": f"{owner}/{repo}",
            "base_commit": base,
            "head_commit": head,
            "url": compare_data.get("html_url"),
            "total_commits": len(commits),
            "total_files_changed": len(file_changes),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "commits": commits,
            "file_changes": file_changes
        }
        
        # Prepare a clear diff representation for the LLM to analyze
        diff_representation = f"Changes between {base} and {head} in {owner}/{repo}:\n\n"
        
        # Add commit information
        diff_representation += "COMMITS:\n"
        for i, commit in enumerate(commits):
            diff_representation += f"{i+1}. {commit['sha']} - {commit['message']} (by {commit['author']} on {commit['date']})\n"
        
        diff_representation += f"\nTOTAL: {len(file_changes)} files changed, {total_additions} additions, {total_deletions} deletions\n\n"
        
        # Add file changes with patches
        for file in file_changes:
            filename = file["filename"]
            status = file["status"].upper()
            changes = f"+{file['additions']}/-{file['deletions']}"
            
            diff_representation += f"FILE: {filename} ({status}) {changes}\n"
            if "patch" in file:
                diff_representation += "```diff\n" + file["patch"] + "\n```\n\n"
            else:
                diff_representation += "No diff available for this file\n\n"
        
        print(f"[DEBUG] Prepared diff representation for LLM analysis (length: {len(diff_representation)})")
        
        # If the diff is too large, truncate it but keep the structure
        if len(diff_representation) > 8000:  # Reasonable limit for most LLMs
            print(f"[DEBUG] Diff too large ({len(diff_representation)} chars), truncating")
            truncated_diff = diff_representation[:2000]  # Start with header
            
            # Add a sample of file changes
            file_samples = 0
            for file in file_changes:
                if "patch" in file and file_samples < 3:  # Limit to 3 files with patches
                    filename = file["filename"]
                    status = file["status"].upper()
                    changes = f"+{file['additions']}/-{file['deletions']}"
                    
                    file_diff = f"FILE: {filename} ({status}) {changes}\n"
                    patch = file["patch"]
                    
                    # Truncate each patch if needed
                    if len(patch) > 1000:
                        patch = patch[:1000] + "\n... (truncated)"
                    
                    file_diff += "```diff\n" + patch + "\n```\n\n"
                    truncated_diff += file_diff
                    file_samples += 1
            
            truncated_diff += f"\n... (truncated, {len(file_changes) - file_samples} more files changed)"
            diff_representation = truncated_diff
            print(f"[DEBUG] Truncated diff to {len(diff_representation)} chars")
        
        # Now we can use our agent's LLM to analyze the changes
        print(f"[DEBUG] Asking LLM to analyze the changes")
        
        # Define a prompt for the LLM to analyze the diff
        analysis_prompt = (
            "Below is a git diff showing changes between two commits or branches. "
            "Please analyze these changes and provide a clear, concise explanation of what was done. "
            "Focus on the purpose and implications of the changes, not just the mechanical differences. "
            "If there are multiple logical changes, group them together. "
            "Identify patterns, bug fixes, new features, refactorings, or other notable aspects.\n\n"
            f"{diff_representation}"
        )
        
        # Return both the raw diff data and the LLM's analysis
        return {
            "status": "success",
            "summary": summary,
            "analysis_prompt": analysis_prompt,
            # Note: The actual analysis will be done at the agent level since we need to call the LLM
            # The agent will use this prompt to get the explanation
        }
    
    except Exception as e:
        print(f"[DEBUG] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Error analyzing changes: {str(e)}"
        }

def get_spinnaker_deployments(pipeline_config_id: Optional[str] = None, application: Optional[str] = None, limit: int = 5, statuses: Optional[str] = None) -> dict:
    """Get deployment details from Spinnaker.
    
    Args:
        pipeline_config_id (str, optional): Filter by pipeline configuration ID.
        application (str, optional): Filter by application name.
        limit (int, optional): Maximum number of deployments to return (default 5).
        statuses (str, optional): Filter by execution statuses (comma-separated).
        
    Returns:
        dict: Status and deployment information or error message.
    """
    print(f"[DEBUG] get_spinnaker_deployments called with: pipeline_config_id={pipeline_config_id}, application={application}, limit={limit}, statuses={statuses}")
    
    # Get Spinnaker API configuration 
    spinnaker_api_url = os.environ.get("SPINNAKER_API_URL", "https://deploy-api.razorpay.com")
    
    # Set up headers based on the curl command
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://deploy.razorpay.com",
        "Referer": "https://deploy.razorpay.com/",
        "X-RateLimit-App": "deck",
        "rzpctx-dev-serve-user": "credithub",
        "x-pg-service": "pg-router"
    }
    
    # Set up cookies (you may need to refresh these periodically)
    cookies = {
        "ab_user_id": "cc071617-5d9f-413d-a4ce-e2228cd525e3",
        "additional-cache-params": "{\"isMobile\":false,\"isSafari\":false,\"countryCode\":\"IN\",\"isTransparentVideoNotSupported\":false,\"isBot\":false,\"host\":\"razorpay.com\",\"previewAsset\":\"\",\"preferenceCountryCode\":null}",
        "visit_time_stamp": "1743562123017",
        "hubspotutk": "9058335a8122f23e0b5945ac65e25009",
        "__hssrc": "1",
        "ajs_user_id": "GIoRgiKEkV89t8",
        "ajs_anonymous_id": "c5d2a126-d94e-436c-a73f-63f0b54097e6",
        "_fbp": "fb.1.1743760311133.326922372503041647",
        "__hstc": "227020674.9058335a8122f23e0b5945ac65e25009.1743760310869.1744369287914.1744619954364.15",
        "analytics_session_id": "1744619955081",
        "analytics_session_id.last_access": "1744619955081",
        "SESSION": "MTkxZmRlNzYtYTIxYi00M2U0LWIzYTgtMGFiN2I3NzY0MGZj"
    }
    
    try:
        # Determine which endpoint to use based on parameters provided
        params = {"limit": limit}
        
        if statuses:
            params["statuses"] = statuses
            
        if pipeline_config_id:
            # Use executions endpoint with pipelineConfigIds filter
            endpoint = f"{spinnaker_api_url}/executions"
            params["pipelineConfigIds"] = pipeline_config_id
            
        elif application:
            # Use application-specific pipelines endpoint
            endpoint = f"{spinnaker_api_url}/applications/{application}/pipelines"
        else:
            # Default to general executions endpoint
            endpoint = f"{spinnaker_api_url}/executions"
        
        print(f"[DEBUG] Requesting Spinnaker deployments from: {endpoint}")
        print(f"[DEBUG] With params: {params}")
        
        response = requests.get(endpoint, headers=headers, cookies=cookies, params=params)
        status_code = response.status_code
        print(f"[DEBUG] Spinnaker API response status: {status_code}")
        
        if status_code != 200:
            error_msg = f"Spinnaker API error: Status {status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details.get('message', '')}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            return {
                "status": "error",
                "error_message": error_msg
            }
        
        # Parse the response data
        deployments_data = response.json()
        
        if not isinstance(deployments_data, list):
            return {
                "status": "error",
                "error_message": "Unexpected response format from Spinnaker API"
            }
        
        # Format the deployments for easier consumption
        deployments = []
        for execution in deployments_data:
            try:
                # Extract key deployment information
                deployment_info = {
                    "id": execution.get("id"),
                    "name": execution.get("name"),
                    "application": execution.get("application"),
                    "status": execution.get("status"),
                    "start_time": execution.get("startTime"),
                    "end_time": execution.get("endTime"),
                    "build_time": (execution.get("endTime", 0) - execution.get("startTime", 0)) / 1000 if execution.get("endTime") else None,
                    "pipeline_config_id": execution.get("pipelineConfigId"),
                    "trigger": execution.get("trigger", {})
                }
                
                # Add stages information (simplified)
                stages = []
                for stage in execution.get("stages", []):
                    stage_info = {
                        "id": stage.get("id"),
                        "name": stage.get("name"),
                        "status": stage.get("status"),
                        "type": stage.get("type"),
                        "refId": stage.get("refId"),
                        "start_time": stage.get("startTime"),
                        "end_time": stage.get("endTime")
                    }
                    stages.append(stage_info)
                
                deployment_info["stages"] = stages
                deployments.append(deployment_info)
                
            except Exception as e:
                print(f"[DEBUG] Error processing deployment: {str(e)}")
                continue
        
        return {
            "status": "success",
            "total_count": len(deployments_data),
            "deployments": deployments
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Spinnaker API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

def get_spinnaker_deployment_details(execution_id: str) -> dict:
    """Get detailed information about a specific Spinnaker deployment/execution.
    
    Args:
        execution_id (str): The ID of the Spinnaker execution/deployment.
        
    Returns:
        dict: Status and detailed deployment information or error message.
    """
    print(f"[DEBUG] get_spinnaker_deployment_details called with execution_id: {execution_id}")
    
    # Get Spinnaker API configuration
    spinnaker_api_url = os.environ.get("SPINNAKER_API_URL", "https://deploy-api.razorpay.com")
    
    # Set up headers based on the curl command
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://deploy.razorpay.com",
        "Referer": "https://deploy.razorpay.com/",
        "X-RateLimit-App": "deck",
        "rzpctx-dev-serve-user": "credithub",
        "x-pg-service": "pg-router"
    }
    
    # Set up cookies (you may need to refresh these periodically)
    cookies = {
        "ab_user_id": "cc071617-5d9f-413d-a4ce-e2228cd525e3",
        "additional-cache-params": "{\"isMobile\":false,\"isSafari\":false,\"countryCode\":\"IN\",\"isTransparentVideoNotSupported\":false,\"isBot\":false,\"host\":\"razorpay.com\",\"previewAsset\":\"\",\"preferenceCountryCode\":null}",
        "visit_time_stamp": "1743562123017",
        "hubspotutk": "9058335a8122f23e0b5945ac65e25009",
        "__hssrc": "1",
        "ajs_user_id": "GIoRgiKEkV89t8",
        "ajs_anonymous_id": "c5d2a126-d94e-436c-a73f-63f0b54097e6",
        "_fbp": "fb.1.1743760311133.326922372503041647",
        "__hstc": "227020674.9058335a8122f23e0b5945ac65e25009.1743760310869.1744369287914.1744619954364.15",
        "analytics_session_id": "1744619955081",
        "analytics_session_id.last_access": "1744619955081",
        "SESSION": "MTkxZmRlNzYtYTIxYi00M2U0LWIzYTgtMGFiN2I3NzY0MGZj"
    }
    
    try:
        # Get execution details using the pipelines endpoint
        execution_url = f"{spinnaker_api_url}/pipelines/{execution_id}"
        print(f"[DEBUG] Requesting execution details from: {execution_url}")
        
        response = requests.get(execution_url, headers=headers, cookies=cookies)
        status_code = response.status_code
        print(f"[DEBUG] Execution details response status: {status_code}")
        
        if status_code != 200:
            return {
                "status": "error",
                "error_message": f"Execution not found or error: Status {status_code}"
            }
        
        execution_data = response.json()
        
        # Format the detailed execution information
        execution_details = {
            "id": execution_data.get("id"),
            "name": execution_data.get("name"),
            "application": execution_data.get("application"),
            "status": execution_data.get("status"),
            "start_time": execution_data.get("startTime"),
            "end_time": execution_data.get("endTime"),
            "build_time_ms": execution_data.get("buildTime"),
            "canceled": execution_data.get("canceled", False),
            "cancellation_reason": execution_data.get("canceledBy"),
            "pipeline_config_id": execution_data.get("pipelineConfigId"),
            "authentication": execution_data.get("authentication", {}),
            "trigger": execution_data.get("trigger", {})
        }
        
        # Process stages in detail
        stages = []
        for stage in execution_data.get("stages", []):
            stage_info = {
                "id": stage.get("id"),
                "refId": stage.get("refId"),
                "name": stage.get("name"),
                "status": stage.get("status"),
                "type": stage.get("type"),
                "start_time": stage.get("startTime"),
                "end_time": stage.get("endTime"),
                "requisite_stage_ref_ids": stage.get("requisiteStageRefIds", []),
                "sync_stage": stage.get("syntheticStageOwner", ""),
                "context": {k: v for k, v in stage.get("context", {}).items() if not isinstance(v, dict) and not isinstance(v, list)}
            }
            
            # Extract task details if available
            tasks = []
            for task in stage.get("tasks", []):
                task_info = {
                    "id": task.get("id"),
                    "name": task.get("name"),
                    "status": task.get("status"),
                    "start_time": task.get("startTime"),
                    "end_time": task.get("endTime"),
                    "message": task.get("message", "")
                }
                tasks.append(task_info)
            
            if tasks:
                stage_info["tasks"] = tasks
            
            stages.append(stage_info)
        
        execution_details["stages"] = stages
        
        # Get pipeline logs if available
        try:
            logs_url = f"{spinnaker_api_url}/pipelines/{execution_id}/logs"
            logs_response = requests.get(logs_url, headers=headers, cookies=cookies)
            if logs_response.status_code == 200:
                logs_data = logs_response.json()
                execution_details["logs"] = logs_data
        except Exception as e:
            print(f"[DEBUG] Error fetching logs: {str(e)}")
        
        return {
            "status": "success",
            "execution_details": execution_details
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Spinnaker API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

def cancel_spinnaker_pipeline(execution_id: str, reason: str = "") -> dict:
    """Cancel a running Spinnaker pipeline execution.
    
    Args:
        execution_id (str): The ID of the pipeline execution to cancel.
        reason (str, optional): Reason for cancellation.
        
    Returns:
        dict: Status of the cancellation operation.
    """
    print(f"[DEBUG] cancel_spinnaker_pipeline called with: execution_id={execution_id}, reason={reason}")
    
    # Get Spinnaker API configuration
    spinnaker_api_url = os.environ.get("SPINNAKER_API_URL", "https://deploy-api.razorpay.com")
    
    # Set up headers based on the curl command
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://deploy.razorpay.com",
        "Referer": "https://deploy.razorpay.com/",
        "X-RateLimit-App": "deck",
        "rzpctx-dev-serve-user": "credithub",
        "x-pg-service": "pg-router",
        "Content-Type": "application/json"
    }
    
    # Set up cookies (you may need to refresh these periodically)
    cookies = {
        "ab_user_id": "cc071617-5d9f-413d-a4ce-e2228cd525e3",
        "additional-cache-params": "{\"isMobile\":false,\"isSafari\":false,\"countryCode\":\"IN\",\"isTransparentVideoNotSupported\":false,\"isBot\":false,\"host\":\"razorpay.com\",\"previewAsset\":\"\",\"preferenceCountryCode\":null}",
        "visit_time_stamp": "1743562123017",
        "hubspotutk": "9058335a8122f23e0b5945ac65e25009",
        "__hssrc": "1",
        "ajs_user_id": "GIoRgiKEkV89t8",
        "ajs_anonymous_id": "c5d2a126-d94e-436c-a73f-63f0b54097e6",
        "_fbp": "fb.1.1743760311133.326922372503041647",
        "__hstc": "227020674.9058335a8122f23e0b5945ac65e25009.1743760310869.1744369287914.1744619954364.15",
        "analytics_session_id": "1744619955081",
        "analytics_session_id.last_access": "1744619955081",
        "SESSION": "MTkxZmRlNzYtYTIxYi00M2U0LWIzYTgtMGFiN2I3NzY0MGZj"
    }
    
    try:
        # Cancel pipeline execution
        params = {}
        if reason:
            params["reason"] = reason
            
        cancel_url = f"{spinnaker_api_url}/pipelines/{execution_id}/cancel"
        print(f"[DEBUG] Requesting pipeline cancellation from: {cancel_url}")
        
        response = requests.put(cancel_url, headers=headers, cookies=cookies, params=params)
        status_code = response.status_code
        print(f"[DEBUG] Pipeline cancellation response status: {status_code}")
        
        if status_code not in [200, 202]:
            error_msg = f"Pipeline cancellation error: Status {status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details.get('message', '')}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            return {
                "status": "error",
                "error_message": error_msg
            }
        
        return {
            "status": "success",
            "message": f"Pipeline execution {execution_id} has been canceled successfully"
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Spinnaker API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

def pause_resume_spinnaker_pipeline(execution_id: str, action: str) -> dict:
    """Pause or resume a Spinnaker pipeline execution.
    
    Args:
        execution_id (str): The ID of the pipeline execution to pause or resume.
        action (str): Either 'pause' or 'resume'.
        
    Returns:
        dict: Status of the operation.
    """
    if action not in ['pause', 'resume']:
        return {
            "status": "error",
            "error_message": f"Invalid action: {action}. Must be 'pause' or 'resume'."
        }
    
    print(f"[DEBUG] pause_resume_spinnaker_pipeline called with: execution_id={execution_id}, action={action}")
    
    # Get Spinnaker API configuration
    spinnaker_api_url = os.environ.get("SPINNAKER_API_URL", "https://deploy-api.razorpay.com")
    
    # Set up headers based on the curl command
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://deploy.razorpay.com",
        "Referer": "https://deploy.razorpay.com/",
        "X-RateLimit-App": "deck",
        "rzpctx-dev-serve-user": "credithub",
        "x-pg-service": "pg-router",
        "Content-Type": "application/json"
    }
    
    # Set up cookies (you may need to refresh these periodically)
    cookies = {
        "ab_user_id": "cc071617-5d9f-413d-a4ce-e2228cd525e3",
        "additional-cache-params": "{\"isMobile\":false,\"isSafari\":false,\"countryCode\":\"IN\",\"isTransparentVideoNotSupported\":false,\"isBot\":false,\"host\":\"razorpay.com\",\"previewAsset\":\"\",\"preferenceCountryCode\":null}",
        "visit_time_stamp": "1743562123017",
        "hubspotutk": "9058335a8122f23e0b5945ac65e25009",
        "__hssrc": "1",
        "ajs_user_id": "GIoRgiKEkV89t8",
        "ajs_anonymous_id": "c5d2a126-d94e-436c-a73f-63f0b54097e6",
        "_fbp": "fb.1.1743760311133.326922372503041647",
        "__hstc": "227020674.9058335a8122f23e0b5945ac65e25009.1743760310869.1744369287914.1744619954364.15",
        "analytics_session_id": "1744619955081",
        "analytics_session_id.last_access": "1744619955081",
        "SESSION": "MTkxZmRlNzYtYTIxYi00M2U0LWIzYTgtMGFiN2I3NzY0MGZj"
    }
    
    try:
        # Pause or resume pipeline execution
        action_url = f"{spinnaker_api_url}/pipelines/{execution_id}/{action}"
        print(f"[DEBUG] Requesting pipeline {action} from: {action_url}")
        
        response = requests.put(action_url, headers=headers, cookies=cookies)
        status_code = response.status_code
        print(f"[DEBUG] Pipeline {action} response status: {status_code}")
        
        if status_code not in [200, 202]:
            error_msg = f"Pipeline {action} error: Status {status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details.get('message', '')}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            return {
                "status": "error",
                "error_message": error_msg
            }
        
        return {
            "status": "success",
            "message": f"Pipeline execution {execution_id} has been {action}d successfully"
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Spinnaker API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

def get_spinnaker_applications() -> dict:
    """Get a list of all applications in Spinnaker.
    
    Returns:
        dict: Status and list of applications or error message.
    """
    print(f"[DEBUG] get_spinnaker_applications called")
    
    # Get Spinnaker API configuration
    spinnaker_api_url = os.environ.get("SPINNAKER_API_URL", "https://deploy-api.razorpay.com")
    
    # Set up headers based on the curl command
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://deploy.razorpay.com",
        "Referer": "https://deploy.razorpay.com/",
        "X-RateLimit-App": "deck",
        "rzpctx-dev-serve-user": "credithub",
        "x-pg-service": "pg-router"
    }
    
    # Set up cookies (you may need to refresh these periodically)
    cookies = {
        "ab_user_id": "cc071617-5d9f-413d-a4ce-e2228cd525e3",
        "additional-cache-params": "{\"isMobile\":false,\"isSafari\":false,\"countryCode\":\"IN\",\"isTransparentVideoNotSupported\":false,\"isBot\":false,\"host\":\"razorpay.com\",\"previewAsset\":\"\",\"preferenceCountryCode\":null}",
        "visit_time_stamp": "1743562123017",
        "hubspotutk": "9058335a8122f23e0b5945ac65e25009",
        "__hssrc": "1",
        "ajs_user_id": "GIoRgiKEkV89t8",
        "ajs_anonymous_id": "c5d2a126-d94e-436c-a73f-63f0b54097e6",
        "_fbp": "fb.1.1743760311133.326922372503041647",
        "__hstc": "227020674.9058335a8122f23e0b5945ac65e25009.1743760310869.1744369287914.1744619954364.15",
        "analytics_session_id": "1744619955081",
        "analytics_session_id.last_access": "1744619955081",
        "SESSION": "MTkxZmRlNzYtYTIxYi00M2U0LWIzYTgtMGFiN2I3NzY0MGZj"
    }
    
    try:
        # Get all applications
        apps_url = f"{spinnaker_api_url}/applications"
        print(f"[DEBUG] Requesting applications from: {apps_url}")
        
        response = requests.get(apps_url, headers=headers, cookies=cookies)
        status_code = response.status_code
        print(f"[DEBUG] Applications response status: {status_code}")
        
        if status_code != 200:
            error_msg = f"Applications request error: Status {status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details.get('message', '')}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            return {
                "status": "error",
                "error_message": error_msg
            }
        
        applications_data = response.json()
        
        if not isinstance(applications_data, list):
            return {
                "status": "error",
                "error_message": "Unexpected response format from Spinnaker API"
            }
        
        # Format application info
        applications = []
        for app in applications_data:
            app_info = {
                "name": app.get("name"),
                "email": app.get("email"),
                "description": app.get("description", ""),
                "cloud_providers": app.get("cloudProviders", []),
                "created_at": app.get("createTs", ""),
                "updated_at": app.get("updateTs", "")
            }
            applications.append(app_info)
        
        return {
            "status": "success",
            "total_count": len(applications),
            "applications": applications
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Spinnaker API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

def get_spinnaker_pipeline_configs(application: str) -> dict:
    """Get pipeline configurations for a specific application.
    
    Args:
        application (str): The name of the application.
        
    Returns:
        dict: Status and pipeline configurations or error message.
    """
    print(f"[DEBUG] get_spinnaker_pipeline_configs called with application: {application}")
    
    # Get Spinnaker API configuration
    spinnaker_api_url = os.environ.get("SPINNAKER_API_URL", "https://deploy-api.razorpay.com")
    
    # Set up headers based on the curl command
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://deploy.razorpay.com",
        "Referer": "https://deploy.razorpay.com/",
        "X-RateLimit-App": "deck",
        "rzpctx-dev-serve-user": "credithub",
        "x-pg-service": "pg-router"
    }
    
    # Set up cookies (you may need to refresh these periodically)
    cookies = {
        "ab_user_id": "cc071617-5d9f-413d-a4ce-e2228cd525e3",
        "additional-cache-params": "{\"isMobile\":false,\"isSafari\":false,\"countryCode\":\"IN\",\"isTransparentVideoNotSupported\":false,\"isBot\":false,\"host\":\"razorpay.com\",\"previewAsset\":\"\",\"preferenceCountryCode\":null}",
        "visit_time_stamp": "1743562123017",
        "hubspotutk": "9058335a8122f23e0b5945ac65e25009",
        "__hssrc": "1",
        "ajs_user_id": "GIoRgiKEkV89t8",
        "ajs_anonymous_id": "c5d2a126-d94e-436c-a73f-63f0b54097e6",
        "_fbp": "fb.1.1743760311133.326922372503041647",
        "__hstc": "227020674.9058335a8122f23e0b5945ac65e25009.1743760310869.1744369287914.1744619954364.15",
        "analytics_session_id": "1744619955081",
        "analytics_session_id.last_access": "1744619955081",
        "SESSION": "MTkxZmRlNzYtYTIxYi00M2U0LWIzYTgtMGFiN2I3NzY0MGZj"
    }
    
    try:
        # Get pipeline configs for application
        configs_url = f"{spinnaker_api_url}/applications/{application}/pipelineConfigs"
        print(f"[DEBUG] Requesting pipeline configs from: {configs_url}")
        
        response = requests.get(configs_url, headers=headers, cookies=cookies)
        status_code = response.status_code
        print(f"[DEBUG] Pipeline configs response status: {status_code}")
        
        if status_code != 200:
            error_msg = f"Pipeline configs request error: Status {status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details.get('message', '')}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            return {
                "status": "error",
                "error_message": error_msg
            }
        
        configs_data = response.json()
        
        if not isinstance(configs_data, list):
            return {
                "status": "error",
                "error_message": "Unexpected response format from Spinnaker API"
            }
        
        # Format pipeline config info
        pipeline_configs = []
        for config in configs_data:
            # Extract basic pipeline config info
            config_info = {
                "id": config.get("id"),
                "name": config.get("name"),
                "application": config.get("application"),
                "description": config.get("description", ""),
                "updated_at": config.get("updateTs"),
                "created_at": config.get("createTs"),
                "index": config.get("index"),
                "disabled": config.get("disabled", False),
                "keep_waiting_pipelines": config.get("keepWaitingPipelines", False),
                "limit_concurrent": config.get("limitConcurrent", False),
                "trigger_types": [trigger.get("type") for trigger in config.get("triggers", [])]
            }
            
            # Get number of stages
            stages = config.get("stages", [])
            config_info["stage_count"] = len(stages)
            config_info["stage_types"] = [stage.get("type") for stage in stages]
            
            pipeline_configs.append(config_info)
        
        return {
            "status": "success",
            "total_count": len(pipeline_configs),
            "pipeline_configs": pipeline_configs
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Spinnaker API error: {str(e)}"
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }

# Create a LiteLLM client for Azure OpenAI
azure_llm = LiteLlm(
    model="azure/razor-genie",  # Format: "azure/deployment_name"
    api_base="https://docs-search-aus-east.openai.azure.com",
    api_key="e2c8b179c3a24127a0ab3b38509fb9b1",
    api_version="2023-05-15"
)

# Create the agent with all tools including GitHub tools
root_agent = Agent(
    name="multi_tool_agent",
    model=azure_llm,
    description=(
        "Agent to answer questions about the time, weather, GitHub information, and Spinnaker deployments."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time, weather, "
        "provide information from GitHub, and fetch deployment details from Spinnaker. "
        "For GitHub queries, you can search for repositories, users, issues, get detailed information "
        "about specific repositories, compare changes between commits, and explain code changes in natural language. "
        "For Spinnaker queries, you can retrieve deployment information and execution details from "
        "the Razorpay deployment system."
    ),
    tools=[
        get_weather, 
        get_current_time, 
        github_search, 
        github_repo_info, 
        github_compare_commits, 
        github_explain_changes,
        get_spinnaker_deployments,
        get_spinnaker_deployment_details,
        cancel_spinnaker_pipeline,
        pause_resume_spinnaker_pipeline,
        get_spinnaker_applications,
        get_spinnaker_pipeline_configs
    ],
)

async def analyze_github_changes(owner: str, repo: str, base: str, head: str) -> dict:
    """Analyze and explain code changes between two commits in a GitHub repository.
    
    Args:
        owner (str): The owner (user or organization) of the repository.
        repo (str): The name of the repository.
        base (str): The base commit SHA, branch, or tag to compare from.
        head (str): The head commit SHA, branch, or tag to compare to.
        
    Returns:
        dict: Status and explanation of changes or error message.
    """
    # First, get the raw diff data and prepare the analysis prompt
    diff_result = github_explain_changes(owner, repo, base, head)
    
    if diff_result["status"] != "success":
        return diff_result
    
    try:
        # Use the agent's LLM to analyze the changes
        analysis_prompt = diff_result["analysis_prompt"]
        
        # For debugging, we'll directly use the LiteLLM client from our agent
        print(f"[DEBUG] Sending analysis prompt to LLM")
        
        # The analysis will be done using the agent's LLM client
        # In a regular case, you would call root_agent.generate_content(analysis_prompt)
        # But for an explanation of code changes, it's better to use a direct call to the LLM
        # with a specialized prompt
        
        response = azure_llm.generate_content(analysis_prompt)
        
        print(f"[DEBUG] Received analysis from LLM")
        
        # Combine the summary data with the analysis
        result = {
            "status": "success",
            "summary": diff_result["summary"],
            "explanation": response
        }
        
        return result
    
    except Exception as e:
        print(f"[DEBUG] Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "error_message": f"Error during LLM analysis: {str(e)}"
        }