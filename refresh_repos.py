import os
import json
import sqlite3
import urllib.parse
import urllib.request

USERNAME = os.environ.get("GITHUB_USER", "FrostBlazeX")
TOKEN = os.environ.get("GITHUB_TOKEN")
INCLUDE_FORKS = os.environ.get("INCLUDE_FORKS", "0") == "1"
DB_FILE = os.environ.get("DB_FILE", "github.db")


def _get(url: str):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "repo-refresh-script",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_all_repos():
    # With a token, /user/repos sees private repos too; without one we can only
    # read the user's public repos.
    base = "https://api.github.com/user/repos" if TOKEN \
        else f"https://api.github.com/users/{USERNAME}/repos"
    repos, page = [], 1
    while True:
        params = {"per_page": 100, "page": page}
        if TOKEN:
            params["affiliation"] = "owner"   # repos you own, incl. private
        batch = _get(f"{base}?{urllib.parse.urlencode(params)}")
        repos.extend(batch)
        if len(batch) < 100:      # last page
            break
        page += 1
    return repos


def main():
    repos = fetch_all_repos()
    if not INCLUDE_FORKS:
        repos = [r for r in repos if not r.get("fork")]

    conn = sqlite3.connect(DB_FILE)
    conn.execute("DROP TABLE IF EXISTS repos")
    conn.execute(
        "CREATE TABLE repos (name TEXT, stars INTEGER, language TEXT, url TEXT)")
    conn.executemany(
        "INSERT INTO repos (name, stars, language, url) VALUES (?, ?, ?, ?)",
        [(r["name"], r.get("stargazers_count", 0), r.get("language"),
          r["html_url"]) for r in repos],
    )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0]
    rows = conn.execute(
        "SELECT COALESCE(language, 'Unknown') AS lang, COUNT(*) AS n "
        "FROM repos GROUP BY lang ORDER BY n DESC").fetchall()
    conn.close()

    print(f"Wrote {total} repos to {DB_FILE} "
          f"(forks {'included' if INCLUDE_FORKS else 'skipped'}, "
          f"{'with token' if TOKEN else 'public only'})")
    for lang, n in rows:
        print(f"  {lang}: {n}")


if __name__ == "__main__":
    main()
