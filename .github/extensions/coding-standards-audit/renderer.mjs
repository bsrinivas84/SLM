export const AUDIT = [
    {
        priority: "high",
        category: "Project structure",
        title: "Create an importable Python package",
        evidence:
            "Numbered directories and files containing dots cannot be imported using normal package syntax.",
        recommendation:
            "Move reusable code under src/slm and keep numbered chapter demonstrations under examples.",
    },
    {
        priority: "high",
        category: "Automation",
        title: "Add formatting and linting",
        evidence:
            "No pyproject.toml, Ruff, Black, EditorConfig, or pre-commit configuration is present.",
        recommendation:
            "Configure Ruff as the single formatter, import sorter, and linter, then enforce it in CI.",
    },
    {
        priority: "high",
        category: "Automation",
        title: "Run quality checks in CI",
        evidence: "The repository has no GitHub Actions workflows.",
        recommendation:
            "Run formatting checks, linting, and unit tests for every pull request.",
    },
    {
        priority: "high",
        category: "Runtime design",
        title: "Remove import-time side effects",
        evidence:
            "Many modules read files, construct models, set random seeds, or print output when imported.",
        recommendation:
            "Keep reusable modules side-effect free and move demonstrations into main functions.",
    },
    {
        priority: "medium",
        category: "Naming",
        title: "Standardize file and directory names",
        evidence:
            "Names mix dots, underscores, PascalCase, numeric prefixes, and misspellings such as Compacy and Multhead.",
        recommendation:
            "Use lowercase snake_case modules and lowercase package directories without dots.",
    },
    {
        priority: "medium",
        category: "Python style",
        title: "Use four-space indentation consistently",
        evidence: "The audit found 884 source lines containing tab indentation.",
        recommendation:
            "Use four spaces and let the formatter normalize indentation and trailing whitespace.",
        action: "fix_indentation",
    },
    {
        priority: "medium",
        category: "Typing and documentation",
        title: "Expand type hints and API documentation",
        evidence:
            "Only 23 of 64 functions are fully annotated; modules and classes have no docstrings.",
        recommendation:
            "Annotate public functions and document tensor shapes, return values, and exceptions.",
    },
    {
        priority: "medium",
        category: "Dependencies",
        title: "Make environments reproducible",
        evidence:
            "TensorFlow is unpinned, development dependencies are not separated, and no Python version or lockfile is defined.",
        recommendation:
            "Declare supported Python versions and dependency groups in pyproject.toml and commit a lockfile.",
    },
    {
        priority: "medium",
        category: "Architecture",
        title: "Identify canonical implementations",
        evidence:
            "Tokenizer, attention, GPT, and generation implementations are duplicated across chapters.",
        recommendation:
            "Keep one tested canonical implementation and label chapter files as educational examples.",
    },
    {
        priority: "low",
        category: "Repository policy",
        title: "Document contribution and review standards",
        evidence:
            "CONTRIBUTING, SECURITY, issue templates, and pull request templates are absent.",
        recommendation:
            "Document coding style, test expectations, commit conventions, and security reporting.",
    },
    {
        priority: "low",
        category: "Artifacts",
        title: "Define generated and binary artifact policy",
        evidence:
            "Generated CSV splits, archives, PDF, DOCX, plots, and images are tracked without a stated policy.",
        recommendation:
            "Identify source versus generated assets and use Git LFS where repository growth requires it.",
    },
];

export function summarizeAudit(audit, focus = "all") {
    const findings =
        focus === "all" ? audit : audit.filter((item) => item.priority === focus);
    return {
        total: findings.length,
        byPriority: {
            high: findings.filter((item) => item.priority === "high").length,
            medium: findings.filter((item) => item.priority === "medium").length,
            low: findings.filter((item) => item.priority === "low").length,
        },
        titles: findings.map((item) => item.title),
    };
}

export function renderHtml(instanceId) {
    return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coding Standards Audit</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--background-color-default, #fff);
      color: var(--text-color-default, #1f2328);
      font-family: var(--font-sans, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
      font-size: var(--text-body-medium, 14px);
      line-height: var(--leading-body-medium, 20px);
    }
    main { max-width: 1080px; margin: 0 auto; padding: 24px; }
    header { display: flex; gap: 20px; justify-content: space-between; align-items: start; }
    h1 {
      margin: 0 0 6px;
      font-size: var(--text-title-large, 26px);
      line-height: var(--leading-title-large, 32px);
      font-weight: var(--font-weight-semibold, 600);
    }
    .muted { color: var(--text-color-muted, #59636e); margin: 0; }
    .score {
      min-width: 112px;
      padding: 14px;
      border: 1px solid var(--border-color-default, #d1d9e0);
      border-radius: 10px;
      text-align: center;
    }
    .score strong { display: block; font-size: 28px; line-height: 32px; }
    .filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 24px 0 16px; }
    button {
      border: 1px solid var(--border-color-default, #d1d9e0);
      border-radius: 7px;
      padding: 7px 12px;
      background: var(--background-color-default, #fff);
      color: var(--text-color-default, #1f2328);
      font: inherit;
      cursor: pointer;
    }
    button:hover { background: var(--n-1, #f6f8fa); }
    button:focus-visible { outline: 2px solid var(--color-focus-outline, #0969da); }
    button[aria-pressed="true"] {
      background: var(--b-9, #0969da);
      border-color: var(--b-9, #0969da);
      color: var(--color-white, #fff);
    }
    .summary { margin: 0 0 16px; color: var(--text-color-muted, #59636e); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }
    article {
      border: 1px solid var(--border-color-default, #d1d9e0);
      border-radius: 10px;
      padding: 16px;
    }
    article h2 { margin: 8px 0; font-size: 16px; line-height: 22px; }
    article p { margin: 8px 0 0; }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: var(--font-weight-semibold, 600);
      text-transform: uppercase;
    }
    .high { color: var(--true-color-red, #cf222e); background: var(--true-color-red-muted, #ffebe9); }
    .medium { color: #9a6700; background: #fff8c5; }
    .low { color: var(--true-color-blue, #0969da); background: var(--true-color-blue-muted, #ddf4ff); }
    .category { margin-left: 8px; color: var(--text-color-muted, #59636e); font-size: 12px; }
    .recommendation { border-top: 1px solid var(--border-color-default, #d1d9e0); padding-top: 9px; }
    .action { margin-top: 12px; }
    .action button { border-color: var(--true-color-blue, #0969da); }
    .resolved { color: #1a7f37; font-weight: var(--font-weight-semibold, 600); }
    .result {
      display: none;
      margin: 0 0 16px;
      padding: 10px 12px;
      border: 1px solid var(--border-color-default, #d1d9e0);
      border-radius: 7px;
    }
    .result.visible { display: block; }
    code { font-family: var(--font-mono, Consolas, monospace); font-size: var(--text-code-inline, 12px); }
    @media (max-width: 560px) {
      main { padding: 16px; }
      header { display: block; }
      .score { margin-top: 16px; width: 112px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Coding Standards Audit</h1>
        <p class="muted">Prioritized engineering standards missing from this repository.</p>
      </div>
      <div class="score"><strong id="count">-</strong><span>findings</span></div>
    </header>
    <nav class="filters" aria-label="Filter findings">
      <button data-focus="all" aria-pressed="false">All</button>
      <button data-focus="high" aria-pressed="false">High</button>
      <button data-focus="medium" aria-pressed="false">Medium</button>
      <button data-focus="low" aria-pressed="false">Low</button>
    </nav>
    <div class="result" id="result" role="status" aria-live="polite"></div>
    <p class="summary" id="summary" aria-live="polite"></p>
    <section class="grid" id="findings"></section>
  </main>
  <script>
    const instanceId = ${JSON.stringify(instanceId)};
    let audit = [];
    let focus = "all";

    function render() {
      const visible = focus === "all"
        ? audit
        : audit.filter((item) => item.priority === focus);
      document.getElementById("count").textContent = visible.length;
      document.getElementById("summary").textContent =
        focus === "all"
          ? "Showing all findings, ordered by adoption priority."
          : "Showing " + focus + "-priority findings.";
      document.querySelectorAll("[data-focus]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.focus === focus));
      });
      document.getElementById("findings").replaceChildren(
        ...visible.map((item) => {
          const article = document.createElement("article");
          article.innerHTML =
            '<span class="badge ' + item.priority + '">' + item.priority + '</span>' +
            '<span class="category">' + item.category + '</span>' +
            '<h2>' + item.title + '</h2>' +
            '<p>' + item.evidence + '</p>' +
            '<p class="recommendation"><strong>Recommendation:</strong> ' +
            item.recommendation + '</p>' +
            (item.action === "fix_indentation" && !item.resolved
              ? '<div class="action"><button data-action="fix_indentation">Fix indentation</button></div>'
              : item.action === "fix_indentation"
                ? '<div class="action resolved">Resolved</div>'
              : '');
          return article;
        }),
      );
    }

    async function selectFocus(nextFocus) {
      const response = await fetch("/api/focus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ focus: nextFocus }),
      });
      if (!response.ok) throw new Error("Unable to update filter");
      focus = nextFocus;
      render();
    }

    async function loadAudit() {
      const response = await fetch("/api/audit");
      const data = await response.json();
      audit = data.audit;
      focus = data.focus;
      render();
    }

    document.querySelector(".filters").addEventListener("click", (event) => {
      const button = event.target.closest("[data-focus]");
      if (button) selectFocus(button.dataset.focus);
    });

    document.getElementById("findings").addEventListener("click", async (event) => {
      const button = event.target.closest('[data-action="fix_indentation"]');
      if (!button) return;
      if (!window.confirm(
        "Replace leading tabs with four spaces in src Python files? " +
        "Each file will be changed only if its parsed AST remains identical.",
      )) return;

      const result = document.getElementById("result");
      button.disabled = true;
      result.classList.add("visible");
      result.textContent = "Checking and updating Python indentation...";
      try {
        const response = await fetch("/api/fix-indentation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirmed: true }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Indentation update failed");
        result.textContent = data.changedFiles === 0
          ? "No indentation changes were needed."
          : "Updated " + data.changedFiles + " files and " +
            data.changedLines + " lines. " +
            (data.repairedFiles
              ? data.repairedFiles + " files with invalid mixed indentation now compile."
              : "All parsed ASTs remained unchanged.");
        await loadAudit();
      } catch (error) {
        result.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });

    loadAudit();
  </script>
</body>
</html>`;
}
