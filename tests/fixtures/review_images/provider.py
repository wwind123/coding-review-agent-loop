"""Inert canned HTTP provider; used only inside an isolated network namespace."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

provider = None
reasoning_messages = False
tool_call = False
requests = []


def codex_events():
    response = {"id": "resp_test", "object": "response", "model": "provider-substitute"}
    events = [("response.created", {"response": {**response, "status": "in_progress", "output": []}})]
    output = []
    if tool_call and len(requests) == 1:
        item = {"id": "fc_test", "type": "function_call", "call_id": "call_test",
                "name": "exec_command", "arguments": json.dumps({"cmd": "printf TOOL_CALLED"})}
        events += [("response.output_item.added", {"output_index": 0, "item": {**item, "arguments": ""}}),
                   ("response.output_item.done", {"output_index": 0, "item": item})]
        output.append(item)
    if reasoning_messages:
        item = {"id": "rs_test", "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Inspecting layout."}]}
        events += [("response.output_item.added", {"output_index": 0, "item": {**item, "summary": []}}),
                   ("response.reasoning_summary_text.delta", {"item_id": "rs_test", "output_index": 0,
                                                            "summary_index": 0, "delta": "Inspecting layout."}),
                   ("response.output_item.done", {"output_index": 0, "item": item})]
        output.append(item)
    for text in (["PREAMBLE", "VISUAL_OK"] if reasoning_messages else ["VISUAL_OK"]):
        index = len(output)
        item = {"id": f"msg_{index}", "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        events += [("response.output_item.added", {"output_index": index, "item": {**item, "content": []}}),
                   ("response.output_text.delta", {"item_id": item["id"], "output_index": index,
                                                  "content_index": 0, "delta": text}),
                   ("response.output_item.done", {"output_index": index, "item": item})]
        output.append(item)
    events.append(("response.completed", {"response": {**response, "status": "completed", "output": output,
                                                      "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}))
    return events


def claude_events(model):
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}})]
    blocks = []
    if reasoning_messages:
        blocks += [{"type": "thinking", "thinking": "Inspecting layout.", "signature": "fixture-signature"},
                   {"type": "text", "text": "PREAMBLE"}]
    blocks.append({"type": "text", "text": "VISUAL_OK"})
    for index, block in enumerate(blocks):
        kind = block["type"]
        events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                               "content_block": {"type": kind, kind: ""}}))
        events.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                               "delta": {"type": kind + "_delta", kind: block[kind]}}))
        if kind == "thinking":
            events.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "signature_delta", "signature": block["signature"]}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events += [("message_delta", {"type": "message_delta", "delta": {
        "stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}}),
               ("message_stop", {"type": "message_stop"})]
    return events


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(data)
        events = codex_events() if provider == "codex" else claude_events(data.get("model", "claude-sonnet-4-6"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for kind, event in events:
            event["type"] = kind
            self.wfile.write(("event: " + kind + "\ndata: " + json.dumps(event) + "\n\n").encode())
