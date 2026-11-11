# Contract fixtures

These files are the contract between the General Context SDK and the platform: the task, benchmark, runs, collected calls and status shapes each side writes and reads. They are copied byte-identical into the platform repo's `server/contract/`, and both repos test against them, so a change here is a change to the contract and must land in both places.

`voice.json` is a collected batch from a voice agent: two `openai-realtime` calls (a tool call, then a spoken reply with `voice` timing on the assistant item) and one `openai-chat` audio call.
