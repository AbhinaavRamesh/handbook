# Chatbot System Design

> **Design production-ready conversational AI systems with multi-turn context, personas, and human handoff**

---

## Learning Objectives

By the end of this module, you will be able to:

- **Design end-to-end chatbot architectures** for customer support, sales, and general assistance
- **Implement multi-turn conversation management** with context windows and summarization
- **Create persona systems** that maintain consistent chatbot personality and behavior
- **Build human handoff mechanisms** for seamless escalation to live agents
- **Address safety concerns** specific to conversational AI systems

---

## Chatbot Architecture Overview

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WebChat[Web Chat Widget]
        MobileChat[Mobile App]
        Voice[Voice Interface]
        SMS[SMS/WhatsApp]
    end

    subgraph "Channel Gateway"
        Gateway[Unified Gateway]
        SessionMgr[Session Manager]
        ChannelAdapter[Channel Adapters]
    end

    subgraph "Conversation Engine"
        Router[Intent Router]
        Context[Context Manager]
        Persona[Persona Engine]
        Dialog[Dialog Manager]
    end

    subgraph "AI Services"
        NLU[NLU Service]
        RAG[RAG Service]
        LLM[LLM Service]
        Safety[Safety Service]
    end

    subgraph "Business Logic"
        Actions[Action Executor]
        Workflow[Workflow Engine]
        Integration[Integration Hub]
    end

    subgraph "Data Layer"
        ConvStore[(Conversation Store)]
        UserStore[(User Profiles)]
        KnowledgeBase[(Knowledge Base)]
        Analytics[(Analytics)]
    end

    subgraph "Human Handoff"
        Queue[Agent Queue]
        AgentUI[Agent Interface]
        Routing[Smart Routing]
    end

    WebChat --> Gateway
    MobileChat --> Gateway
    Voice --> Gateway
    SMS --> Gateway

    Gateway --> SessionMgr --> ChannelAdapter
    ChannelAdapter --> Router

    Router --> Context
    Router --> Persona
    Router --> Dialog

    Dialog --> NLU
    Dialog --> RAG
    Dialog --> LLM
    Dialog --> Safety

    Dialog --> Actions
    Actions --> Workflow
    Workflow --> Integration

    Context --> ConvStore
    Dialog --> UserStore
    RAG --> KnowledgeBase
    Dialog --> Analytics

    Dialog -->|Escalation| Queue
    Queue --> Routing --> AgentUI
```

### Core Components

| Component | Purpose | Key Considerations |
|-----------|---------|-------------------|
| **Channel Gateway** | Unified entry point for all channels | Protocol translation, rate limiting |
| **Session Manager** | Track user sessions across interactions | TTL, persistence, multi-device |
| **Context Manager** | Maintain conversation state | Window size, summarization |
| **Persona Engine** | Consistent chatbot personality | Tone, boundaries, brand voice |
| **Dialog Manager** | Orchestrate conversation flow | State machine, error recovery |
| **Action Executor** | Execute business actions | Idempotency, rollback |

---

## Multi-Turn Conversation Management

### The Context Challenge

```mermaid
graph TB
    subgraph "Conversation Flow"
        U1[User: Hi, I need help<br/>with my order]
        B1[Bot: I'd be happy to help!<br/>What's your order number?]
        U2[User: It's #12345]
        B2[Bot: I found order #12345.<br/>What's the issue?]
        U3[User: I want to return it]
        B3[Bot: I can help with that.<br/>What's the reason?]
        U4[User: It doesn't fit]
        B4[Bot: Got it. I've initiated<br/>a return for sizing issues.]
    end

    U1 --> B1 --> U2 --> B2 --> U3 --> B3 --> U4 --> B4

    subgraph "Context Required"
        C1[Order #12345]
        C2[Return request]
        C3[Sizing issue]
        C4[User preferences]
    end

    B2 -.- C1
    B3 -.- C2
    B4 -.- C3
    B4 -.- C4
```

### Context Management Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Full History** | Send entire conversation | Maximum context | Token cost, latency |
| **Sliding Window** | Last N messages | Bounded cost | Loses early context |
| **Summarization** | Summarize older messages | Efficient, comprehensive | Summary quality varies |
| **Hybrid** | Recent full + summarized older | Balance | Implementation complexity |
| **Memory Retrieval** | Retrieve relevant past exchanges | Efficient for long conversations | Retrieval quality |

### Hybrid Context Architecture

```mermaid
graph TB
    subgraph "Conversation History"
        Old[Old Messages<br/>Turn 1-20]
        Recent[Recent Messages<br/>Turn 21-30]
        Current[Current Turn<br/>Turn 31]
    end

    subgraph "Context Processing"
        Summarizer[Summarizer LLM]
        Window[Sliding Window]
        Merger[Context Merger]
    end

    subgraph "Final Context"
        SystemPrompt[System Prompt]
        Summary[Conversation Summary]
        RecentHistory[Recent History<br/>Last 10 turns]
        CurrentMsg[Current Message]
    end

    Old --> Summarizer --> Summary
    Recent --> Window --> RecentHistory
    Current --> CurrentMsg

    SystemPrompt --> Merger
    Summary --> Merger
    RecentHistory --> Merger
    CurrentMsg --> Merger

    Merger --> LLM[LLM]
```

### Context Window Sizing

::: info Token Budget Allocation
For a 4K token context window, allocate:
- System prompt: ~500 tokens (12.5%)
- Summary: ~500 tokens (12.5%)
- Recent history: ~2000 tokens (50%)
- Current turn + response space: ~1000 tokens (25%)
:::

| Context Size | Messages | Summary Strategy | Use Case |
|--------------|----------|------------------|----------|
| **4K tokens** | ~10-15 | Aggressive summarization | Simple support |
| **8K tokens** | ~25-30 | Periodic summarization | Standard chatbot |
| **32K tokens** | ~100-120 | Minimal summarization | Complex workflows |
| **128K+ tokens** | ~500+ | Full history possible | Long-running sessions |

### Conversation State Machine

```mermaid
stateDiagram-v2
    [*] --> Greeting: New Session
    Greeting --> Understanding: User Message
    Understanding --> Clarifying: Need More Info
    Clarifying --> Understanding: User Clarifies
    Understanding --> Executing: Intent Clear
    Executing --> Confirming: Action Taken
    Confirming --> Understanding: Continue
    Confirming --> Closing: Complete

    Understanding --> Escalating: Cannot Handle
    Executing --> Escalating: Failure
    Escalating --> AgentHandoff: Transfer

    Closing --> [*]: Session End
    AgentHandoff --> [*]: Agent Takes Over
```

---

## Persona Design

### Persona Architecture

```mermaid
graph TB
    subgraph "Persona Definition"
        Identity[Identity<br/>Name, Role, Background]
        Voice[Voice<br/>Tone, Style, Vocabulary]
        Boundaries[Boundaries<br/>Can/Cannot Do]
        Knowledge[Knowledge<br/>Expertise Areas]
    end

    subgraph "Persona Application"
        SystemPrompt[System Prompt]
        FewShot[Few-Shot Examples]
        Guardrails[Response Guardrails]
    end

    subgraph "Consistency Checks"
        StyleChecker[Style Consistency]
        BoundaryChecker[Boundary Enforcement]
        ToneAnalyzer[Tone Analysis]
    end

    Identity --> SystemPrompt
    Voice --> SystemPrompt
    Voice --> FewShot
    Boundaries --> Guardrails
    Knowledge --> SystemPrompt

    SystemPrompt --> LLM[LLM]
    FewShot --> LLM
    LLM --> Response[Response]
    Response --> StyleChecker
    Response --> BoundaryChecker
    Response --> ToneAnalyzer
```

### Persona Template

```yaml
persona:
  name: "Alex"
  role: "Customer Support Specialist"
  company: "TechCorp"

  voice:
    tone: "friendly, professional, helpful"
    style: "conversational but efficient"
    vocabulary: "avoid jargon, explain technical terms"

  boundaries:
    can_do:
      - Answer product questions
      - Process returns and exchanges
      - Check order status
      - Provide troubleshooting help
    cannot_do:
      - Discuss competitor products
      - Share internal policies
      - Make promises about future features
      - Discuss pricing negotiations

  behavior:
    greeting: "Hi there! I'm Alex from TechCorp support."
    farewell: "Thanks for chatting! Have a great day!"
    uncertainty: "I want to make sure I give you accurate info..."
    escalation: "Let me connect you with a specialist..."
```

### Persona Consistency Techniques

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **System Prompt** | Define persona in system instructions | Always include persona details |
| **Few-Shot Examples** | Show example conversations | 3-5 representative exchanges |
| **Style Guide** | Post-process for consistency | Regex patterns, style classifiers |
| **Tone Analysis** | Monitor response tone | Sentiment analysis, custom models |
| **A/B Testing** | Test persona variations | Measure user satisfaction |

---

## Human Handoff System

### Handoff Architecture

```mermaid
graph TB
    subgraph "Detection Layer"
        Intent[Intent Detection]
        Sentiment[Sentiment Analysis]
        Frustration[Frustration Detection]
        Complexity[Complexity Assessment]
        Explicit[Explicit Request]
    end

    subgraph "Decision Engine"
        Scorer[Handoff Scorer]
        Policy[Policy Engine]
        Capacity[Capacity Checker]
    end

    subgraph "Queue Management"
        Priority[Priority Queue]
        Skills[Skill-Based Routing]
        SLA[SLA Monitor]
    end

    subgraph "Agent Interface"
        Summary[Conversation Summary]
        Context[Full Context]
        Suggestions[Suggested Responses]
        Actions[Quick Actions]
    end

    subgraph "Transition"
        Warmup[Warm Handoff]
        Notify[User Notification]
        Transfer[Session Transfer]
    end

    Intent --> Scorer
    Sentiment --> Scorer
    Frustration --> Scorer
    Complexity --> Scorer
    Explicit --> Scorer

    Scorer --> Policy --> Capacity
    Capacity -->|Available| Priority
    Capacity -->|Unavailable| Fallback[Callback/Email]

    Priority --> Skills --> SLA --> Warmup
    Warmup --> Notify --> Transfer

    Transfer --> Summary
    Transfer --> Context
    Transfer --> Suggestions
    Transfer --> Actions
```

### Handoff Triggers

| Trigger Type | Detection Method | Priority |
|--------------|------------------|----------|
| **Explicit Request** | "Talk to human", "Agent please" | Immediate |
| **High Frustration** | Sentiment score, repeated questions | High |
| **Complex Issue** | Multi-domain, policy questions | Medium |
| **Safety Concern** | Threat detection, urgent issues | Immediate |
| **Bot Failure** | Repeated misunderstanding | High |
| **High-Value Customer** | User tier, purchase history | Medium |

### Handoff Scoring Model

```python
def calculate_handoff_score(conversation):
    score = 0

    # Explicit request (immediate)
    if detect_explicit_handoff_request(conversation.last_message):
        return 100

    # Sentiment analysis (0-30 points)
    sentiment = analyze_sentiment(conversation.recent_messages)
    score += (1 - sentiment) * 30  # Lower sentiment = higher score

    # Frustration indicators (0-25 points)
    frustration = detect_frustration(conversation)
    score += frustration * 25

    # Bot confidence (0-20 points)
    confidence = conversation.last_bot_confidence
    score += (1 - confidence) * 20

    # Conversation length (0-15 points)
    turns = len(conversation.turns)
    score += min(turns / 20, 1) * 15  # Max at 20 turns

    # Repeated issues (0-10 points)
    repetitions = count_repeated_intents(conversation)
    score += min(repetitions / 3, 1) * 10

    return min(score, 100)
```

### Warm Handoff Flow

```mermaid
sequenceDiagram
    participant U as User
    participant B as Bot
    participant Q as Queue
    participant A as Agent

    U->>B: "I need to speak to someone"
    B->>B: Detect handoff trigger
    B->>Q: Request agent
    Q->>Q: Check availability
    Q->>A: Assign conversation
    B->>U: "Connecting you with an agent..."
    A->>A: Review summary
    Q->>A: Transfer session
    A->>U: "Hi, I'm Sarah. I've reviewed your issue about..."

    Note over B,A: Conversation summary includes:<br/>- Key issues<br/>- Actions taken<br/>- Customer sentiment
```

---

## Safety in Conversational AI

### Conversational Safety Architecture

```mermaid
graph TB
    subgraph "Input Safety"
        Input[User Message]
        PIIDetect[PII Detection]
        ToxicDetect[Toxicity Detection]
        Injection[Injection Detection]
    end

    subgraph "Processing"
        Safe[Safe Input]
        LLM[LLM Processing]
        Raw[Raw Response]
    end

    subgraph "Output Safety"
        OutputMod[Output Moderation]
        Hallucination[Hallucination Check]
        Boundary[Boundary Check]
        Redact[PII Redaction]
    end

    subgraph "Response"
        Final[Final Response]
        Fallback[Fallback Response]
        Escalate[Escalate to Human]
    end

    Input --> PIIDetect --> ToxicDetect --> Injection
    Injection -->|Clean| Safe --> LLM --> Raw
    Injection -->|Flagged| Escalate

    Raw --> OutputMod --> Hallucination --> Boundary --> Redact --> Final

    OutputMod -->|Blocked| Fallback
    Hallucination -->|High Risk| Fallback
    Boundary -->|Violated| Fallback
```

### Chatbot-Specific Safety Concerns

| Concern | Risk | Mitigation |
|---------|------|------------|
| **Manipulation** | User manipulating bot to give harmful info | Strict persona boundaries, detection |
| **Impersonation** | Bot pretending to be human | Clear bot disclosure |
| **Over-Promise** | Making commitments bot can't keep | Action boundary enforcement |
| **Emotional Manipulation** | Exploiting user emotions | Sentiment monitoring, handoff |
| **Data Collection** | Inadvertent data gathering | PII detection, minimal storage |
| **Confidentiality Breach** | Revealing other users' data | Strict data isolation |

---

## Interview Q&A

### Q1: How do you handle context in long conversations?

**Answer:**
"I implement a hybrid context management strategy:

1. **Sliding window**: Keep the last N turns (e.g., 10) in full detail
2. **Progressive summarization**: Summarize older context periodically
3. **Key entity extraction**: Maintain structured data (order numbers, user preferences)
4. **Memory retrieval**: For very long conversations, retrieve relevant past exchanges

The architecture:
```
[System Prompt] + [Summary of turns 1-20] + [Full turns 21-30] + [Current turn]
```

I trigger summarization when context exceeds 70% of the budget, keeping buffer for response generation. For critical entities, I store them separately and always inject into context."

### Q2: How do you ensure consistent chatbot persona?

**Answer:**
"I implement persona consistency at multiple levels:

1. **System prompt**: Detailed persona definition with name, role, voice, boundaries
2. **Few-shot examples**: 3-5 representative conversations showing desired behavior
3. **Post-processing**: Style classifiers to catch tone drift, regex for forbidden phrases
4. **Continuous monitoring**: Track persona consistency metrics, A/B test variations

For boundary enforcement, I use a two-pass approach:
- First pass: Generate response
- Second pass: Validate against persona constraints (can/cannot do lists)

If validation fails, regenerate with stricter instructions or use a fallback response."

### Q3: Design the human handoff system for a support chatbot.

**Answer:**
"I design a multi-signal handoff system:

**Detection**: Score conversations on:
- Explicit requests (immediate trigger)
- Sentiment trajectory (frustration detection)
- Bot confidence scores
- Conversation length and repetition
- Issue complexity classification

**Routing**: Once handoff triggers:
1. Check agent availability and skills
2. Generate conversation summary
3. Queue with priority based on customer tier and issue urgency
4. Notify user with expected wait time

**Transition**: Warm handoff process:
1. Bot: 'Connecting you with a specialist...'
2. Agent receives summary + full context
3. Agent: 'Hi, I'm Sarah. I see you're having trouble with...'

**Fallback**: If agents unavailable:
- Offer callback scheduling
- Provide email option with case number
- Estimate response time

**Agent Tools**: Give agents:
- AI-suggested responses
- Sentiment analysis
- Quick action buttons
- Knowledge base integration"

### Q4: How would you handle a user trying to manipulate the chatbot?

**Answer:**
"I implement defense in depth:

**Detection**:
- Prompt injection patterns (ignoring instructions, roleplay attempts)
- Jailbreak attempts (pretend games, hypothetical scenarios)
- Social engineering (claiming to be admin, urgency tactics)

**Prevention**:
- Strong system prompt with explicit boundaries
- Input sanitization and classification
- Output validation against allowed topics
- Behavioral anomaly detection

**Response Strategy**:
1. Soft redirect: 'I'm here to help with X, Y, Z. How can I assist?'
2. Firm boundary: 'I can't help with that, but I can...'
3. Handoff: For persistent attempts, offer human agent
4. Session termination: For clearly malicious behavior

I also log manipulation attempts for analysis and model improvement, while being careful not to create a cat-and-mouse game that trains better attacks."

---

## Trade-off Analysis

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Context Strategy** | Full history (accurate) | Summarized (efficient) | Hybrid: recent full + summarized old |
| **Persona Enforcement** | Hard rules (safe) | Soft guidance (natural) | Hard for boundaries, soft for style |
| **Handoff Timing** | Early (safe) | Late (cost-effective) | Dynamic scoring with user preference |
| **Channel Unification** | Single codebase (maintainable) | Channel-specific (optimized) | Unified core + channel adapters |
| **State Storage** | Client-side (simple) | Server-side (rich) | Server-side with client session ID |

---

## System Design Checklist

Before finalizing your chatbot design, verify:

- [ ] Multi-channel support architecture defined
- [ ] Context management strategy with token budgets
- [ ] Persona definition with boundaries and voice
- [ ] Human handoff triggers and routing logic
- [ ] Input/output safety guardrails
- [ ] Session management and persistence
- [ ] Integration with business systems (CRM, ticketing)
- [ ] Monitoring and evaluation metrics
- [ ] Fallback strategies for edge cases
- [ ] Scalability plan for peak loads

---

## Navigation

| Previous | Next |
|----------|------|
| [Design Framework](./design-framework) | [Enterprise RAG](./enterprise-rag) |

---

## Sources

- Jurafsky, D. & Martin, J.H. (2024). *Speech and Language Processing, Chapter 24: Chatbots*. https://web.stanford.edu/~jurafsky/slp3/
- Microsoft. (2024). *Bot Framework Documentation*. https://docs.microsoft.com/en-us/azure/bot-service/
- Rasa. (2024). *Conversational AI Best Practices*. https://rasa.com/docs/
- Anthropic. (2024). *Prompt Caching*. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- OpenAI. (2024). *GPT Best Practices for Conversational AI*. https://platform.openai.com/docs/guides/gpt-best-practices
