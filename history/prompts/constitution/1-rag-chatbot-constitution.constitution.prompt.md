---
id: 1
title: rag-chatbot-constitution
stage: constitution
date: '2025-12-09'
model: 'Qwen'
feature: 'RAG Chatbot'
branch: 'main'
user: 'tayyaba'
command: '/sp.constitution'
labels:
  - 'constitution'
  - 'rag'
  - 'chatbot'
  - 'physical-ai'
links:
  - 'null'
files:
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\.specify\memory\constitution.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\.specify\templates\plan-template.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\.specify\templates\spec-template.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\.specify\templates\tasks-template.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\README.md
tests: []
prompt: |
  ---
  description: Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync
  ---

  ## User Input

  ```text
  $ARGUMENTS
  ```

  You **MUST** consider the user input before proceeding (if not empty).

  ## Outline

  You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

  Follow this execution flow:

  1. Load the existing constitution template at `.specify/memory/constitution.md`.
     - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
     **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

  2. Collect/derive values for placeholders:
     - If user input (conversation) supplies a value, use it.
     - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
     - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
     - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
       - MAJOR: Backward incompatible governance/principle removals or redefinitions.
       - MINOR: New principle/section added or materially expanded guidance.
       - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
     - If version bump type ambiguous, propose reasoning before finalizing.

  3. Draft the updated constitution content:
     - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
     - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
     - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
     - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

  4. Consistency propagation checklist (convert prior checklist into active validations):
     - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
     - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
     - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
     - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
     - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

  5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
     - Version change: old → new
     - List of modified principles (old title → new title if renamed)
     - Added sections
     - Removed sections
     - Templates requiring updates (✅ updated / ⚠ pending) with file paths
     - Follow-up TODOs if any placeholders intentionally deferred.

  6. Validation before final output:
     - No remaining unexplained bracket tokens.
     - Version line matches report.
     - Dates ISO format YYYY-MM-DD.
     - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

  7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

  8. Output a final summary to the user with:
     - New version and bump rationale.
     - Any files flagged for manual follow-up.
     - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

  Formatting & Style Requirements:

  - Use Markdown headings exactly as in the template (do not demote/promote levels).
  - Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
  - Keep a single blank line between sections.
  - Avoid trailing whitespace.

  If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

  If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

  Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

  ---

  As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

  1) Determine Stage
     - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

  2) Generate Title and Determine Routing:
     - Generate Title: 3–7 words (slug for filename)
     - Route is automatically determined by stage:
       - `constitution` → `history/prompts/constitution/`
       - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
       - `general` → `history/prompts/general/`

  3) Create and Fill PHR (Shell first; fallback agent‑native)
     - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
     - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
     - If the script fails:
       - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
       - Allocate an ID; compute the output path based on stage from step 2; write the file
       - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

  4) Validate + report
     - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
     - On failure: warn, don't block. Skip only for `/sp.phr`.


  /sp.constitution 

  You are building a RAG (Retrieval Augmented Generation) chatbot for a Docusaurus-based documentation book. Follow these non-negotiable principles:

  ### Core Principles:
  1. **Context Awareness**: The chatbot MUST understand both general book content AND user-selected text context
  2. **Dual Context Mode**: Support two interaction modes:
     - General Q&A about entire book content
     - Selected-text Q&A where user highlights text and asks questions about it
  3. **Accurate Retrieval**: Always retrieve relevant context before generating responses
  4. **No Hallucination**: Never fabricate information not present in the documentation
  5. **Privacy First**: All API keys and sensitive data must be environment variables
  6. **Performance**: Responses must be under 3 seconds for optimal UX
  7. **Accessibility**: Chatbot UI must be keyboard navigable and screen-reader friendly
  8. **Seamless Integration**: Must not disrupt Docusaurus navigation or reading experience

  ### Technical Constraints:
  - Use OpenAI-compatible Gemini API (not native OpenAI)
  - Use Qdrant Cloud for vector storage
  - Use Neon DB for metadata/chat history storage
  - Must work client-side and server-side appropriately
  - Support markdown rendering in responses
  - Handle code snippets with syntax highlighting

  ### Design Constraints:
  - Animated Tamagotchi-style character as chat interface
  - Purple and black gradient theme throughout
  - Smooth animations (60fps minimum)
  - Mobile-responsive design
  - Dark mode optimized

  ### User Experience Constraints:
  - Chat widget must be collapsible/expandable
  - Must preserve context across page navigation
  - Show typing indicators and loading states
  - Display sources/references for answers
  - Allow users to copy responses
  - Support conversation history
response: |
  I've successfully updated the project constitution to reflect the requirements for the RAG chatbot. Here's what I've done:

  1. **Constitution Update**: I created a new constitution at `.specify/memory/constitution.md` that replaces the old textbook principles with new RAG chatbot-focused principles:
     - Context Awareness
     - Dual Context Mode
     - Accurate Retrieval
     - No Hallucination
     - Privacy First
     - Performance Requirement
     - Accessibility Compliance
     - Seamless Integration
     - Plus additional principles for design and user experience

  2. **Template Updates**: I updated all related templates to ensure consistency:
     - Updated plan-template.md with appropriate constitution checks
     - Updated spec-template.md with RAG-specific functional requirements
     - Updated tasks-template.md with setup tasks for the RAG chatbot

  3. **Documentation Update**: I updated the README.md to reflect the new RAG chatbot features

  4. **Version Update**: The constitution version was bumped from 1.0.0 to 2.0.0 to reflect the major changes in principles and project focus.

  The new constitution follows all the principles you specified, including context awareness, dual context mode, accurate retrieval, no hallucination, privacy-first approach, performance requirements, accessibility compliance, and seamless integration. It also includes the technical constraints using OpenAI-compatible Gemini API, Qdrant Cloud, Neon DB and the design constraints for the animated Tamagotchi-style interface with purple and black theme.

  All templates have been updated to ensure consistency with the new constitutional principles.
outcome: |
  Successfully updated the project constitution to focus on the RAG chatbot feature while maintaining consistency across all related templates. The constitution now properly reflects the non-negotiable principles for the RAG chatbot development.
evaluation: |
  The constitution update was comprehensive, with all related templates updated for consistency. The new principles align with the requirements specified by the user. The version bump from 1.0.0 to 2.0.0 properly reflects the major change in project direction.
---