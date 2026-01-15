import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import { withMermaid } from 'vitepress-plugin-mermaid'
import fs from 'fs'
import path from 'path'

// Check if personal content exists (local dev only)
const hasPersonalContent = fs.existsSync(path.resolve(__dirname, '../behavioural/personal'))

export default withMermaid(defineConfig({
  title: "The Handbook",
  description: "The definitive technical interview resource for ML, AI, and Software Engineering roles",
  base: '/handbook/',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ],

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: 'Home', link: '/' },
      { text: 'SDE Coding', link: '/sde-coding/' },
      {
        text: 'ML Fundamentals',
        items: [
          { text: 'Overview', link: '/ml-fundamentals/' },
          { text: 'Classical ML', link: '/ml-fundamentals/linear-regression' },
          { text: 'Neural Networks', link: '/ml-fundamentals/neural-networks/' },
          { text: 'Transformers', link: '/ml-fundamentals/transformers/' },
          { text: 'Reinforcement Learning', link: '/ml-fundamentals/reinforcement-learning/' },
          { text: 'Interview FAQ', link: '/ml-fundamentals/interview-faq/' }
        ]
      },
      {
        text: 'GenAI Engineering',
        items: [
          { text: 'Overview', link: '/genai/' },
          { text: 'LLM Foundations', link: '/genai/llm-foundations/' },
          { text: 'Prompt Engineering', link: '/genai/prompt-engineering/' },
          { text: 'RAG Systems', link: '/genai/rag-systems/' },
          { text: 'Fine-Tuning', link: '/genai/fine-tuning/' },
          { text: 'Agents & Tools', link: '/genai/agents-and-tools/' },
          { text: 'Evaluation', link: '/genai/evaluation/' },
          { text: 'Safety & Alignment', link: '/genai/safety-and-alignment/' },
          { text: 'LLMOps', link: '/genai/llmops/' },
          { text: 'Multimodal', link: '/genai/multimodal/' },
          { text: 'System Design', link: '/genai/system-design/' }
        ]
      },
      { text: 'ML Coding', link: '/ml-coding/' },
      { text: 'System Design', link: '/ml-design/' },
      { text: 'Behavioural', link: '/behavioural/' }
    ],

    sidebar: {
      '/sde-coding/': [
        {
          text: 'Getting Started',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/sde-coding/' },
            { text: 'Interview Tips', link: '/sde-coding/overview/interview-tips' },
            { text: 'Choosing a Language', link: '/sde-coding/overview/choosing-language' },
            { text: 'Practice Warm-Up', link: '/sde-coding/overview/practice-warmup' }
          ]
        },
        {
          text: 'Fast Track',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/fast-track/' },
            { text: 'How to Answer Questions', link: '/sde-coding/fast-track/answering-questions' },
            { text: 'Arrays & Pointers', link: '/sde-coding/fast-track/arrays-pointers-stacks' },
            { text: 'Binary Search & Heaps', link: '/sde-coding/fast-track/binary-search-heaps' },
            { text: 'Lists, Trees & Tries', link: '/sde-coding/fast-track/lists-trees-tries' },
            { text: 'Backtracking, Graphs & DP', link: '/sde-coding/fast-track/backtracking-graphs-dp' }
          ]
        },
        {
          text: 'Time & Space Complexity',
          collapsed: true,
          items: [
            { text: 'Big O & Time', link: '/sde-coding/complexity/big-o-time' },
            { text: 'Space & Optimization', link: '/sde-coding/complexity/space-optimization' }
          ]
        },
        {
          text: 'Coding Patterns',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/patterns/' },
            { text: 'Two Pointer', link: '/sde-coding/patterns/two-pointer' },
            { text: 'Sliding Window', link: '/sde-coding/patterns/sliding-window' },
            { text: 'Prefix Sum & Cycle Detection', link: '/sde-coding/patterns/prefix-sum-cycle' },
            { text: 'Bit Manipulation & Cyclic Sort', link: '/sde-coding/patterns/bit-manipulation-cyclic' }
          ]
        },
        {
          text: 'Arrays',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/arrays/' },
            { text: 'Three Sum', link: '/sde-coding/arrays/three-sum' },
            { text: 'Merge Intervals & Anagrams', link: '/sde-coding/arrays/merge-intervals-anagrams' },
            { text: 'Product & Spiral', link: '/sde-coding/arrays/product-spiral' },
            { text: 'Subarray & Container', link: '/sde-coding/arrays/subarray-container' },
            { text: 'Task Scheduler & Rain Water', link: '/sde-coding/arrays/task-scheduler-rain' }
          ]
        },
        {
          text: 'Hash Tables',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/hash-tables/' },
            { text: 'Two Sum & Profit', link: '/sde-coding/hash-tables/two-sum-profit' },
            { text: 'Advanced Hash Problems', link: '/sde-coding/hash-tables/advanced-hash' }
          ]
        },
        {
          text: 'Searching & Sorting',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/searching-sorting/' },
            { text: 'Sorting Algorithms', link: '/sde-coding/searching-sorting/sorting-algorithms' },
            { text: 'Binary Search', link: '/sde-coding/searching-sorting/binary-search' },
            { text: 'Rotated Arrays & Koko', link: '/sde-coding/searching-sorting/rotated-koko' },
            { text: 'K-Messed & Duplicates', link: '/sde-coding/searching-sorting/kmessed-duplicates' }
          ]
        },
        {
          text: 'Strings',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/strings/' },
            { text: 'Palindrome & Common Words', link: '/sde-coding/strings/palindrome-common' },
            { text: 'IP Validation & Decrypt', link: '/sde-coding/strings/ip-decrypt' },
            { text: 'Window & Serialize', link: '/sde-coding/strings/window-serialize' },
            { text: 'Substring Problems', link: '/sde-coding/strings/substring-problems' }
          ]
        },
        {
          text: 'Graphs',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/graphs/' },
            { text: 'Graph Search & Friendship', link: '/sde-coding/graphs/search-friendship' },
            { text: 'Islands & Oranges', link: '/sde-coding/graphs/islands-oranges' },
            { text: 'Course Schedule & Edit Distance', link: '/sde-coding/graphs/course-schedule-edit' },
            { text: 'Redundant Connection', link: '/sde-coding/graphs/redundant-connection' }
          ]
        },
        {
          text: 'Trees',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/trees/' },
            { text: 'Balanced, Diameter & Validate', link: '/sde-coding/trees/balanced-diameter-validate' },
            { text: 'Construct Tree & Trie', link: '/sde-coding/trees/construct-trie' },
            { text: 'LCA & BST Operations', link: '/sde-coding/trees/lca-bst-operations' }
          ]
        },
        {
          text: 'Stacks & Queues',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/stacks-queues/' },
            { text: 'Parentheses & Min Stack', link: '/sde-coding/stacks-queues/parentheses-stack' },
            { text: 'Temperatures & Path', link: '/sde-coding/stacks-queues/temperatures-path' }
          ]
        },
        {
          text: 'Linked Lists',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/linked-lists/' },
            { text: 'Reverse & Cycle', link: '/sde-coding/linked-lists/reverse-cycle' },
            { text: 'Merge & LRU Cache', link: '/sde-coding/linked-lists/merge-lru' }
          ]
        },
        {
          text: 'Heaps',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/heaps/' },
            { text: 'Heap Problems', link: '/sde-coding/heaps/heap-problems' }
          ]
        },
        {
          text: 'Recursion & Backtracking',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/recursion-backtracking/' },
            { text: 'Fibonacci, Parentheses & Subsets', link: '/sde-coding/recursion-backtracking/fib-parentheses-subsets' },
            { text: 'Permutations & Combinations', link: '/sde-coding/recursion-backtracking/permutations-combinations' },
            { text: 'Sudoku & Regex', link: '/sde-coding/recursion-backtracking/sudoku-regex' },
            { text: 'Calculator & Misc', link: '/sde-coding/recursion-backtracking/calculator-misc' }
          ]
        },
        {
          text: 'Dynamic Programming',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/dynamic-programming/' },
            { text: 'Climbing Stairs', link: '/sde-coding/dynamic-programming/climbing-stairs' },
            { text: 'House Robber', link: '/sde-coding/dynamic-programming/house-robber' },
            { text: 'Coin Change', link: '/sde-coding/dynamic-programming/coin-change' },
            { text: 'Knapsack', link: '/sde-coding/dynamic-programming/knapsack' },
            { text: 'Longest Increasing Subsequence', link: '/sde-coding/dynamic-programming/longest-increasing-subsequence' },
            { text: 'Longest Common Subsequence', link: '/sde-coding/dynamic-programming/longest-common-subsequence' },
            { text: 'Edit Distance', link: '/sde-coding/dynamic-programming/edit-distance' },
            { text: 'Word Break', link: '/sde-coding/dynamic-programming/word-break' },
            { text: 'Unique Paths', link: '/sde-coding/dynamic-programming/unique-paths' },
            { text: 'Minimum Path Sum', link: '/sde-coding/dynamic-programming/minimum-path-sum' },
            { text: 'Maximum Subarray', link: '/sde-coding/dynamic-programming/maximum-subarray' },
            { text: 'Decode Ways', link: '/sde-coding/dynamic-programming/decode-ways' },
            { text: 'Target Sum', link: '/sde-coding/dynamic-programming/target-sum' },
            { text: 'Partition Equal Subset', link: '/sde-coding/dynamic-programming/partition-equal-subset' },
            { text: 'Matrix Chain', link: '/sde-coding/dynamic-programming/matrix-chain' },
            { text: 'Burst Balloons', link: '/sde-coding/dynamic-programming/burst-balloons' },
            { text: 'Regex Matching', link: '/sde-coding/dynamic-programming/regex-matching' },
            { text: 'Wildcard Matching', link: '/sde-coding/dynamic-programming/wildcard-matching' }
          ]
        }
      ],

      '/ml-fundamentals/': [
        {
          text: 'Getting Started',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/ml-fundamentals/' },
            { text: 'How to Answer', link: '/ml-fundamentals/concepts-overview' }
          ]
        },
        {
          text: 'Classical ML',
          collapsed: false,
          items: [
            {
              text: 'Supervised Learning',
              collapsed: false,
              items: [
                { text: 'Linear Regression', link: '/ml-fundamentals/linear-regression' },
                { text: 'Logistic Regression', link: '/ml-fundamentals/logistic-regression' },
                { text: 'Decision Trees', link: '/ml-fundamentals/decision-trees' },
                { text: 'SVM', link: '/ml-fundamentals/svm' },
                { text: 'KNN', link: '/ml-fundamentals/knn' }
              ]
            },
            {
              text: 'Unsupervised Learning',
              collapsed: false,
              items: [
                { text: 'Clustering', link: '/ml-fundamentals/clustering' }
              ]
            }
          ]
        },
        {
          text: 'Deep Learning',
          collapsed: false,
          items: [
            {
              text: 'Neural Networks',
              collapsed: false,
              items: [
                { text: 'Overview', link: '/ml-fundamentals/neural-networks/' },
                { text: 'Perceptrons & MLPs', link: '/ml-fundamentals/neural-networks/01-perceptrons-mlps' },
                { text: 'Activation Functions', link: '/ml-fundamentals/neural-networks/02-activation-functions' },
                { text: 'Backpropagation', link: '/ml-fundamentals/neural-networks/03-backpropagation' },
                { text: 'Weight Initialization', link: '/ml-fundamentals/neural-networks/04-weight-initialization' },
                { text: 'Normalization', link: '/ml-fundamentals/neural-networks/05-normalization' },
                { text: 'CNNs', link: '/ml-fundamentals/neural-networks/06-cnns' },
                { text: 'RNN, LSTM, GRU', link: '/ml-fundamentals/neural-networks/07-rnns-lstm-gru' },
                { text: 'Autoencoders & VAE', link: '/ml-fundamentals/neural-networks/08-autoencoders-vae' },
                { text: 'GANs', link: '/ml-fundamentals/neural-networks/09-gans' },
                { text: 'Optimizers', link: '/ml-fundamentals/neural-networks/10-optimizers' },
                { text: 'Regularization', link: '/ml-fundamentals/neural-networks/11-regularization' },
                { text: 'Gradient Problems', link: '/ml-fundamentals/neural-networks/12-gradient-problems' }
              ]
            },
            {
              text: 'Transformers & Attention',
              collapsed: true,
              items: [
                { text: 'Overview', link: '/ml-fundamentals/transformers/' },
                { text: 'Attention Fundamentals', link: '/ml-fundamentals/transformers/attention-fundamentals' },
                { text: 'Self-Attention Mechanics', link: '/ml-fundamentals/transformers/self-attention-mechanics' },
                { text: 'Multi-Head Attention', link: '/ml-fundamentals/transformers/multi-head-attention' },
                { text: 'Positional Encoding', link: '/ml-fundamentals/transformers/positional-encoding' },
                { text: 'Encoder Architecture', link: '/ml-fundamentals/transformers/encoder-architecture' },
                { text: 'Decoder Architecture', link: '/ml-fundamentals/transformers/decoder-architecture' },
                { text: 'Training & Optimization', link: '/ml-fundamentals/transformers/training-optimization' },
                { text: 'Model Variants & Scaling', link: '/ml-fundamentals/transformers/model-variants' },
                { text: 'Practical Applications', link: '/ml-fundamentals/transformers/practical-applications' },
                { text: 'Interview Questions', link: '/ml-fundamentals/transformers/interview-questions' }
              ]
            },
            {
              text: 'Reinforcement Learning',
              collapsed: true,
              items: [
                { text: 'Overview', link: '/ml-fundamentals/reinforcement-learning/' },
                { text: 'MDP Foundations', link: '/ml-fundamentals/reinforcement-learning/mdp-foundations' },
                { text: 'Value Functions', link: '/ml-fundamentals/reinforcement-learning/value-functions' },
                { text: 'Exploration vs Exploitation', link: '/ml-fundamentals/reinforcement-learning/exploration-exploitation' },
                { text: 'TD Learning & Q-Learning', link: '/ml-fundamentals/reinforcement-learning/temporal-difference' },
                { text: 'Policy Gradients', link: '/ml-fundamentals/reinforcement-learning/policy-gradients' },
                { text: 'Actor-Critic Methods', link: '/ml-fundamentals/reinforcement-learning/actor-critic' },
                { text: 'Deep RL (DQN)', link: '/ml-fundamentals/reinforcement-learning/deep-rl' },
                { text: 'PPO & Modern Methods', link: '/ml-fundamentals/reinforcement-learning/ppo-modern-methods' },
                { text: 'RLHF & Alignment', link: '/ml-fundamentals/reinforcement-learning/rlhf-alignment' },
                { text: 'Advanced Topics', link: '/ml-fundamentals/reinforcement-learning/advanced-topics' }
              ]
            }
          ]
        },
        {
          text: 'Evaluation',
          collapsed: false,
          items: [
            { text: 'Model Evaluation', link: '/ml-fundamentals/model-evaluation' }
          ]
        },
        {
          text: 'Interview FAQ',
          collapsed: true,
          items: [
            { text: 'FAQ Overview', link: '/ml-fundamentals/interview-faq/' },
            { text: 'Bias-Variance Tradeoff', link: '/ml-fundamentals/interview-faq/core-concepts/bias-variance' },
            { text: 'Regularization (L1/L2)', link: '/ml-fundamentals/interview-faq/core-concepts/regularization' },
            { text: 'Gradient Descent', link: '/ml-fundamentals/interview-faq/optimization/gradient-descent' },
            { text: 'Cross-Validation', link: '/ml-fundamentals/interview-faq/optimization/cross-validation' },
            { text: 'Ensemble Methods', link: '/ml-fundamentals/interview-faq/models/ensemble-methods' },
            { text: 'Hypothesis Testing', link: '/ml-fundamentals/interview-faq/statistics/hypothesis-testing' }
          ]
        }
      ],

      '/ml-coding/': [
        {
          text: 'ML Coding',
          items: [
            { text: 'Overview', link: '/ml-coding/' },
            { text: 'Approach & Patterns', link: '/ml-coding/coding-overview' },
            { text: 'Linear Regression', link: '/ml-coding/linear-regression-implementation' },
            { text: 'Logistic Regression', link: '/ml-coding/logistic-regression-implementation' },
            { text: 'KNN Implementation', link: '/ml-coding/knn-implementation' },
            { text: 'K-Means Implementation', link: '/ml-coding/kmeans-implementation' },
            { text: '2D Convolution', link: '/ml-coding/cnn-filter' },
            { text: 'Practical Problems', link: '/ml-coding/practical-problems' },
            { text: 'Quick Reference', link: '/ml-coding/coding-reference' }
          ]
        }
      ],

      '/ml-design/': [
        {
          text: 'System Design',
          items: [
            { text: 'Overview', link: '/ml-design/' },
            { text: 'ML Domain Round', link: '/ml-design/ml-domain-round' },
            { text: 'Real Interview Questions', link: '/ml-design/real-questions' },
            { text: 'Design Framework', link: '/ml-design/framework' },
            { text: 'Key Phrases', link: '/ml-design/key-phrases' },
            { text: 'Experience Mapping', link: '/ml-design/experience-mapping' }
          ]
        }
      ],

      '/behavioural/': [
        {
          text: 'Leadership & Culture',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/behavioural/' },
            { text: '8 Core Traits', link: '/behavioural/traits' },
            { text: 'STAR Stories', link: '/behavioural/star-stories' },
            { text: 'Common Questions', link: '/behavioural/questions' },
            { text: 'Quick Reference', link: '/behavioural/quick-reference' },
            { text: 'Real Scenarios', link: '/behavioural/real-scenarios' }
          ]
        },
        {
          text: 'Company-Specific',
          collapsed: false,
          items: [
            { text: 'Google', link: '/behavioural/companies/google' },
            { text: 'Meta', link: '/behavioural/companies/meta' }
          ]
        },
        // Personal section only shows in local dev
        ...(hasPersonalContent ? [{
          text: '🔒 My Answers (Local Only)',
          collapsed: false,
          items: [
            { text: 'My STAR Stories', link: '/behavioural/personal/my-stories' },
            { text: 'My Experience Mapping', link: '/behavioural/personal/my-experience' }
          ]
        }] : [])
      ],

      '/genai/': [
        {
          text: 'GenAI Engineering',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/genai/' }
          ]
        },
        {
          text: 'LLM Foundations',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/llm-foundations/' },
            { text: 'How LLMs Work', link: '/genai/llm-foundations/how-llms-work' },
            { text: 'Tokenization Deep Dive', link: '/genai/llm-foundations/tokenization-deep-dive' },
            { text: 'Scaling Laws', link: '/genai/llm-foundations/scaling-laws' },
            { text: 'Model Families', link: '/genai/llm-foundations/model-families' },
            { text: 'Context & Memory', link: '/genai/llm-foundations/context-and-memory' },
            { text: 'Inference Optimization', link: '/genai/llm-foundations/inference-optimization' }
          ]
        },
        {
          text: 'Prompt Engineering',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/prompt-engineering/' },
            { text: 'Prompt Anatomy', link: '/genai/prompt-engineering/prompt-anatomy' },
            { text: 'Few-Shot Learning', link: '/genai/prompt-engineering/few-shot-learning' },
            { text: 'Chain-of-Thought', link: '/genai/prompt-engineering/chain-of-thought' },
            { text: 'Advanced Techniques', link: '/genai/prompt-engineering/advanced-techniques' },
            { text: 'Prompt Optimization', link: '/genai/prompt-engineering/prompt-optimization' },
            { text: 'Prompt Security', link: '/genai/prompt-engineering/prompt-security' },
            { text: 'Prompt Evaluation', link: '/genai/prompt-engineering/prompt-evaluation' }
          ]
        },
        {
          text: 'RAG Systems',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/rag-systems/' },
            { text: 'RAG Architecture', link: '/genai/rag-systems/rag-architecture' },
            { text: 'Embedding Models', link: '/genai/rag-systems/embedding-models' },
            { text: 'Vector Databases', link: '/genai/rag-systems/vector-databases' },
            { text: 'Chunking Strategies', link: '/genai/rag-systems/chunking-strategies' },
            { text: 'Retrieval Optimization', link: '/genai/rag-systems/retrieval-optimization' },
            { text: 'Advanced RAG Patterns', link: '/genai/rag-systems/advanced-rag-patterns' },
            { text: 'RAG Evaluation', link: '/genai/rag-systems/rag-evaluation' }
          ]
        },
        {
          text: 'Fine-Tuning LLMs',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/fine-tuning/' },
            { text: 'When to Fine-Tune', link: '/genai/fine-tuning/when-to-fine-tune' },
            { text: 'Full Fine-Tuning', link: '/genai/fine-tuning/full-fine-tuning' },
            { text: 'PEFT Methods', link: '/genai/fine-tuning/peft-methods' },
            { text: 'Instruction Tuning', link: '/genai/fine-tuning/instruction-tuning' },
            { text: 'RLHF & DPO', link: '/genai/fine-tuning/rlhf-dpo' },
            { text: 'Data Preparation', link: '/genai/fine-tuning/data-preparation' },
            { text: 'Fine-Tuning Evaluation', link: '/genai/fine-tuning/fine-tuning-evaluation' }
          ]
        },
        {
          text: 'Agents & Tool Use',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/agents-and-tools/' },
            { text: 'Agent Architectures', link: '/genai/agents-and-tools/agent-architectures' },
            { text: 'Function Calling', link: '/genai/agents-and-tools/function-calling' },
            { text: 'Multi-Agent Systems', link: '/genai/agents-and-tools/multi-agent-systems' },
            { text: 'Memory & State', link: '/genai/agents-and-tools/memory-and-state' },
            { text: 'Agent Frameworks', link: '/genai/agents-and-tools/agent-frameworks' },
            { text: 'Agent Evaluation', link: '/genai/agents-and-tools/agent-evaluation' }
          ]
        },
        {
          text: 'Evaluation & Benchmarking',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/evaluation/' },
            { text: 'Evaluation Taxonomy', link: '/genai/evaluation/evaluation-taxonomy' },
            { text: 'Automated Metrics', link: '/genai/evaluation/automated-metrics' },
            { text: 'LLM-as-Judge', link: '/genai/evaluation/llm-as-judge' },
            { text: 'Human Evaluation', link: '/genai/evaluation/human-evaluation' },
            { text: 'Hallucination Detection', link: '/genai/evaluation/hallucination-detection' },
            { text: 'Benchmarks', link: '/genai/evaluation/benchmarks' }
          ]
        },
        {
          text: 'Safety & Alignment',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/safety-and-alignment/' },
            { text: 'Safety Fundamentals', link: '/genai/safety-and-alignment/safety-fundamentals' },
            { text: 'Content Filtering', link: '/genai/safety-and-alignment/content-filtering' },
            { text: 'Guardrails', link: '/genai/safety-and-alignment/guardrails' },
            { text: 'Red Teaming', link: '/genai/safety-and-alignment/red-teaming' },
            { text: 'Constitutional AI', link: '/genai/safety-and-alignment/constitutional-ai' },
            { text: 'Responsible AI', link: '/genai/safety-and-alignment/responsible-ai' }
          ]
        },
        {
          text: 'LLMOps & Production',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/llmops/' },
            { text: 'Serving Infrastructure', link: '/genai/llmops/serving-infrastructure' },
            { text: 'Cost Optimization', link: '/genai/llmops/cost-optimization' },
            { text: 'Latency Optimization', link: '/genai/llmops/latency-optimization' },
            { text: 'Caching Strategies', link: '/genai/llmops/caching-strategies' },
            { text: 'Monitoring', link: '/genai/llmops/monitoring' },
            { text: 'Scaling Patterns', link: '/genai/llmops/scaling-patterns' }
          ]
        },
        {
          text: 'Multimodal AI',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/multimodal/' },
            { text: 'Vision-Language Models', link: '/genai/multimodal/vision-language-models' },
            { text: 'Image Generation', link: '/genai/multimodal/image-generation' },
            { text: 'Audio Models', link: '/genai/multimodal/audio-models' },
            { text: 'Video Understanding', link: '/genai/multimodal/video-understanding' },
            { text: 'Multimodal RAG', link: '/genai/multimodal/multimodal-rag' }
          ]
        },
        {
          text: 'GenAI System Design',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/genai/system-design/' },
            { text: 'Design Framework', link: '/genai/system-design/design-framework' },
            { text: 'Chatbot Design', link: '/genai/system-design/chatbot-design' },
            { text: 'Enterprise RAG', link: '/genai/system-design/enterprise-rag' },
            { text: 'Code Assistant', link: '/genai/system-design/code-assistant' },
            { text: 'Content Pipeline', link: '/genai/system-design/content-pipeline' },
            { text: 'Interview Questions', link: '/genai/system-design/interview-questions' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Built for engineers, by engineers',
      copyright: 'The Handbook — Technical Interview Preparation'
    }
  },

  markdown: {
    lineNumbers: true,
    math: true,
    config: (md) => {
      md.use(mathjax3)
    }
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => tag.startsWith('mjx-')
      }
    }
  },

  // Mermaid configuration
  mermaid: {
    // Mermaid theme options
  },
  mermaidPlugin: {
    class: 'mermaid'
  }
}))
