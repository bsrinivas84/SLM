import { createServer } from "node:http";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { promisify } from "node:util";
import {
    CanvasError,
    createCanvas,
    joinSession,
} from "@github/copilot-sdk/extension";

import { AUDIT, renderHtml, summarizeAudit } from "./renderer.mjs";

const servers = new Map();
const validFocusValues = new Set(["all", "high", "medium", "low"]);
const execFileAsync = promisify(execFile);
const extensionDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(extensionDirectory, "../../..");
const indentationFixer = resolve(extensionDirectory, "fix_indentation.py");

function sendJson(response, status, value) {
    response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
    response.end(JSON.stringify(value));
}

async function readJson(request) {
    const chunks = [];
    for await (const chunk of request) {
        chunks.push(chunk);
    }
    const body = Buffer.concat(chunks).toString("utf8");
    return body ? JSON.parse(body) : {};
}

function setFocus(entry, focus) {
    if (!validFocusValues.has(focus)) {
        throw new CanvasError("invalid_focus", `Unknown focus: ${focus}`);
    }
    entry.focus = focus;
    return { focus, summary: summarizeAudit(AUDIT, focus) };
}

async function runIndentationTool({ checkOnly = false, confirmed = false } = {}) {
    if (!checkOnly && confirmed !== true) {
        throw new CanvasError(
            "confirmation_required",
            "Explicit confirmation is required before changing source files.",
        );
    }

    try {
        const { stdout } = await execFileAsync(
            "python",
            [
                indentationFixer,
                resolve(repositoryRoot, "src"),
                ...(checkOnly ? ["--check"] : []),
            ],
            { cwd: repositoryRoot, windowsHide: true },
        );
        return JSON.parse(stdout);
    } catch (error) {
        const detail = error instanceof Error ? error.message : "Unknown error";
        throw new CanvasError(
            "indentation_fix_failed",
            `Unable to fix indentation: ${detail}`,
        );
    }
}

async function currentAudit() {
    const indentation = await runIndentationTool({ checkOnly: true });
    return AUDIT.map((item) => {
        if (item.action !== "fix_indentation") {
            return item;
        }
        const resolved = indentation.changedLines === 0;
        return {
            ...item,
            resolved,
            evidence: resolved
                ? "No actionable leading-tab indentation remains in Python source."
                : `${indentation.changedLines} Python source lines still use leading tabs.`,
        };
    });
}

async function startServer(instanceId, initialFocus) {
    const entry = { focus: initialFocus, server: undefined, url: undefined };
    const server = createServer(async (request, response) => {
        const url = new URL(request.url ?? "/", "http://127.0.0.1");

        try {
            if (request.method === "GET" && url.pathname === "/") {
                response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
                response.end(renderHtml(instanceId));
                return;
            }
            if (request.method === "GET" && url.pathname === "/api/audit") {
                sendJson(response, 200, {
                    audit: await currentAudit(),
                    focus: entry.focus,
                });
                return;
            }
            if (request.method === "POST" && url.pathname === "/api/focus") {
                const input = await readJson(request);
                sendJson(response, 200, setFocus(entry, input.focus));
                return;
            }
            if (
                request.method === "POST" &&
                url.pathname === "/api/fix-indentation"
            ) {
                const input = await readJson(request);
                sendJson(
                    response,
                    200,
                    await runIndentationTool({ confirmed: input.confirmed }),
                );
                return;
            }
            sendJson(response, 404, { error: "Not found" });
        } catch (error) {
            sendJson(response, 400, {
                error: error instanceof Error ? error.message : "Invalid request",
            });
        }
    });

    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    const port = typeof address === "object" && address ? address.port : 0;
    entry.server = server;
    entry.url = `http://127.0.0.1:${port}/`;
    return entry;
}

const session = await joinSession({
    canvases: [
        createCanvas({
            id: "coding-standards-audit",
            displayName: "Coding Standards Audit",
            description:
                "Interactive dashboard of this repository's coding-standard gaps and adoption priorities.",
            inputSchema: {
                type: "object",
                additionalProperties: false,
                properties: {
                    focus: {
                        type: "string",
                        enum: ["all", "high", "medium", "low"],
                    },
                },
            },
            actions: [
                {
                    name: "get_summary",
                    description:
                        "Return counts and titles for the currently selected audit priority.",
                    handler: (ctx) => {
                        const entry = servers.get(ctx.instanceId);
                        if (!entry) {
                            throw new CanvasError(
                                "instance_not_open",
                                "The coding standards canvas is not open.",
                            );
                        }
                        return {
                            focus: entry.focus,
                            summary: summarizeAudit(AUDIT, entry.focus),
                        };
                    },
                },
                {
                    name: "set_focus",
                    description:
                        "Filter the dashboard to all, high, medium, or low priority findings.",
                    inputSchema: {
                        type: "object",
                        additionalProperties: false,
                        required: ["focus"],
                        properties: {
                            focus: {
                                type: "string",
                                enum: ["all", "high", "medium", "low"],
                            },
                        },
                    },
                    handler: (ctx) => {
                        const entry = servers.get(ctx.instanceId);
                        if (!entry) {
                            throw new CanvasError(
                                "instance_not_open",
                                "The coding standards canvas is not open.",
                            );
                        }
                        return setFocus(entry, ctx.input.focus);
                    },
                },
                {
                    name: "fix_indentation",
                    description:
                        "Replace leading tabs with four-space indentation in Python source after explicit confirmation.",
                    inputSchema: {
                        type: "object",
                        additionalProperties: false,
                        required: ["confirmed"],
                        properties: {
                            confirmed: {
                                type: "boolean",
                                enum: [true],
                            },
                        },
                    },
                    handler: (ctx) =>
                        runIndentationTool({ confirmed: ctx.input.confirmed }),
                },
            ],
            open: async (ctx) => {
                const initialFocus = ctx.input?.focus ?? "all";
                let entry = servers.get(ctx.instanceId);
                if (!entry) {
                    entry = await startServer(ctx.instanceId, initialFocus);
                    servers.set(ctx.instanceId, entry);
                } else {
                    setFocus(entry, initialFocus);
                }
                return {
                    title: "Coding Standards Audit",
                    status: `${summarizeAudit(AUDIT, entry.focus).total} findings`,
                    url: entry.url,
                };
            },
            onClose: async (ctx) => {
                const entry = servers.get(ctx.instanceId);
                if (entry) {
                    servers.delete(ctx.instanceId);
                    await new Promise((resolve) => entry.server.close(resolve));
                }
            },
        }),
    ],
});

process.on("SIGTERM", () => {
    for (const entry of servers.values()) {
        entry.server.close();
    }
    servers.clear();
});
