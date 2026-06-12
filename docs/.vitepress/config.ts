import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import taskLists from 'markdown-it-task-lists'
import { withMermaid } from 'vitepress-plugin-mermaid'
import fs from 'fs'
import path from 'path'
import { generateLlmsArtifacts } from './llms'

// Check if personal content exists (local dev only)
const hasPersonalContent = fs.existsSync(path.resolve(__dirname, '../behavioural/personal'))

const SITE_URL = 'https://abhinaavramesh.github.io/handbook/'
const SITE_TITLE = 'The Handbook - Technical Interview Prep'
const SITE_DESCRIPTION = 'The definitive technical interview resource for ML, AI, and Software Engineering roles'

// 'index.md' -> '', 'foo/index.md' -> 'foo/', 'foo/bar.md' -> 'foo/bar.html'
function pageUrl(relativePath: string): string {
  const pagePath = relativePath.endsWith('index.md')
    ? relativePath.slice(0, -'index.md'.length)
    : relativePath.replace(/\.md$/, '.html')
  return SITE_URL + pagePath
}

export default withMermaid(defineConfig({
  title: "The Handbook",
  titleTemplate: ':title | The Handbook',
  description: SITE_DESCRIPTION,
  base: '/handbook/',

  sitemap: {
    hostname: SITE_URL
  },

  head: [
    ['link', { rel: 'icon', href: '/handbook/favicon.ico' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'The Handbook' }],
    ['meta', { property: 'og:image', content: `${SITE_URL}og-image.png` }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:image', content: `${SITE_URL}og-image.png` }],
  ],

  // Per-page canonical URL, social tags and JSON-LD structured data
  transformPageData(pageData) {
    const url = pageUrl(pageData.relativePath)
    const title = pageData.title ? `${pageData.title} | The Handbook` : SITE_TITLE
    const description = pageData.description || SITE_DESCRIPTION
    const isHome = pageData.relativePath === 'index.md'

    const jsonLd = isHome
      ? {
          '@context': 'https://schema.org',
          '@type': 'WebSite',
          name: 'The Handbook',
          url: SITE_URL,
          description: SITE_DESCRIPTION
        }
      : {
          '@context': 'https://schema.org',
          '@type': 'TechArticle',
          headline: pageData.title || 'The Handbook',
          description,
          url,
          inLanguage: 'en',
          isPartOf: { '@type': 'WebSite', name: 'The Handbook', url: SITE_URL }
        }

    pageData.frontmatter.head = [
      ...(pageData.frontmatter.head ?? []),
      ['link', { rel: 'canonical', href: url }],
      ['meta', { property: 'og:url', content: url }],
      ['meta', { property: 'og:title', content: title }],
      ['meta', { property: 'og:description', content: description }],
      ['meta', { name: 'twitter:title', content: title }],
      ['meta', { name: 'twitter:description', content: description }],
      ['script', { type: 'application/ld+json' }, JSON.stringify(jsonLd).replace(/</g, '\\u003c')]
    ]
  },

  buildEnd(siteConfig) {
    generateLlmsArtifacts(siteConfig, SITE_URL)
  },

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
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
          { text: 'Multimodal AI', link: '/genai/multimodal/' },
          { text: 'LLMOps', link: '/genai/llmops/' },
          { text: 'GenAI System Design', link: '/genai/system-design/' }
        ]
      },
      {
        text: 'SDE Coding',
        items: [
          { text: 'Overview', link: '/sde-coding/' },
          { text: 'Java-Specific Pointers', link: '/sde-coding/overview/java-specific-pointers' },
          { text: 'Fast Track', link: '/sde-coding/fast-track/' },
          { text: 'Weekend Sprint', link: '/sde-coding/sprint/' },
          { text: 'Complexity', link: '/sde-coding/complexity/big-o-time' },
          { text: 'Coding Patterns', link: '/sde-coding/patterns/' },
          { text: 'Arrays', link: '/sde-coding/arrays/' },
          { text: 'Strings', link: '/sde-coding/strings/' },
          { text: 'Hash Tables', link: '/sde-coding/hash-tables/' },
          { text: 'Linked Lists', link: '/sde-coding/linked-lists/' },
          { text: 'Stacks & Queues', link: '/sde-coding/stacks-queues/' },
          { text: 'Trees', link: '/sde-coding/trees/' },
          { text: 'Graphs', link: '/sde-coding/graphs/' },
          { text: 'Heaps', link: '/sde-coding/heaps/' },
          { text: 'Searching & Sorting', link: '/sde-coding/searching-sorting/' },
          { text: 'Recursion & Backtracking', link: '/sde-coding/recursion-backtracking/' },
          { text: 'Dynamic Programming', link: '/sde-coding/dynamic-programming/' }
        ]
      },
      {
        text: 'ML Coding',
        items: [
          { text: 'Overview', link: '/ml-coding/' },
          { text: 'Classical ML', items: [
            { text: 'Linear Regression', link: '/ml-coding/linear-regression-implementation' },
            { text: 'Logistic Regression', link: '/ml-coding/logistic-regression-implementation' },
            { text: 'KNN', link: '/ml-coding/knn-implementation' },
            { text: 'K-Means', link: '/ml-coding/kmeans-implementation' },
            { text: 'Decision Tree', link: '/ml-coding/decision-tree-implementation' },
            { text: 'Random Forest', link: '/ml-coding/random-forest' },
            { text: 'Naive Bayes', link: '/ml-coding/naive-bayes' },
            { text: 'SVM', link: '/ml-coding/svm-implementation' },
            { text: 'PCA', link: '/ml-coding/pca-implementation' }
          ]},
          { text: 'Deep Learning', items: [
            { text: 'Neural Network MLP', link: '/ml-coding/neural-network-mlp' },
            { text: 'Softmax & Cross-Entropy', link: '/ml-coding/softmax-crossentropy' },
            { text: 'Self-Attention', link: '/ml-coding/self-attention' },
            { text: 'Batch Norm & Dropout', link: '/ml-coding/batch-norm-dropout' }
          ]}
        ]
      },
      {
        text: 'System Design',
        items: [
          { text: 'Overview', link: '/ml-design/' },
          { text: 'Framework', link: '/ml-design/framework' },
          { text: 'Key Phrases', link: '/ml-design/key-phrases' },
          { text: 'Real Questions', link: '/ml-design/real-questions' }
        ]
      },
      {
        text: 'Behavioural',
        items: [
          { text: 'Overview', link: '/behavioural/' },
          { text: 'STAR Framework', link: '/behavioural/star-framework' },
          { text: 'Questions Bank', link: '/behavioural/questions-bank' },
          { text: 'Story Templates', link: '/behavioural/story-templates' },
          { text: 'Quick Reference', link: '/behavioural/quick-reference' }
        ]
      }
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
            { text: 'Java-Specific Pointers', link: '/sde-coding/overview/java-specific-pointers' },
            { text: 'Practice Warm-Up', link: '/sde-coding/overview/practice-warmup' }
          ]
        },
        {
          text: 'Fast Track',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/fast-track/' },
            { text: 'Week 1: Foundations', link: '/sde-coding/fast-track/week1-foundations' },
            { text: 'Week 2: Advanced', link: '/sde-coding/fast-track/week2-advanced' },
            { text: 'Essential Patterns', link: '/sde-coding/fast-track/essential-patterns' },
            { text: 'Priority Problems', link: '/sde-coding/fast-track/priority-problems' },
            { text: 'How to Answer Questions', link: '/sde-coding/fast-track/answering-questions' },
            { text: 'Arrays & Pointers', link: '/sde-coding/fast-track/arrays-pointers-stacks' },
            { text: 'Binary Search & Heaps', link: '/sde-coding/fast-track/binary-search-heaps' },
            { text: 'Lists, Trees & Tries', link: '/sde-coding/fast-track/lists-trees-tries' },
            { text: 'Backtracking, Graphs & DP', link: '/sde-coding/fast-track/backtracking-graphs-dp' },
            { text: 'Mock Interview Guide', link: '/sde-coding/fast-track/mock-interview-guide' },
            { text: 'Day-of Checklist', link: '/sde-coding/fast-track/day-of-checklist' },
            { text: 'Common Mistakes', link: '/sde-coding/fast-track/common-mistakes' }
          ]
        },
        {
          text: 'Weekend Sprint',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/sprint/' },
            { text: 'Study Schedule', link: '/sde-coding/sprint/schedule' },
            { text: 'Graphs (Tier 1)', link: '/sde-coding/sprint/graphs' },
            { text: 'Trees (Tier 2)', link: '/sde-coding/sprint/trees' },
            { text: 'Sliding Window (Tier 3)', link: '/sde-coding/sprint/sliding-window' },
            { text: 'DP (Tier 4)', link: '/sde-coding/sprint/dp' },
            { text: 'HashMap/Heap/BS (Tier 5)', link: '/sde-coding/sprint/data-structures' },
            { text: 'Strings (Tier 6)', link: '/sde-coding/sprint/strings' },
            { text: 'Code Templates', link: '/sde-coding/sprint/templates' },
            { text: 'Interview Day', link: '/sde-coding/sprint/interview-day' },
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
            { text: 'Two Pointers', link: '/sde-coding/patterns/two-pointers' },
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
            { text: 'Two Sum', link: '/sde-coding/arrays/two-sum' },
            { text: 'Three Sum', link: '/sde-coding/arrays/three-sum' },
            { text: 'Buy Sell Stock', link: '/sde-coding/arrays/buy-sell-stock' },
            { text: 'Contains Duplicate', link: '/sde-coding/arrays/contains-duplicate' },
            { text: 'Product Except Self', link: '/sde-coding/arrays/product-except-self' },
            { text: 'Maximum Subarray', link: '/sde-coding/arrays/maximum-subarray' },
            { text: 'Merge Intervals', link: '/sde-coding/arrays/merge-intervals' },
            { text: 'Container Water', link: '/sde-coding/arrays/container-water' },
            { text: 'Trapping Rain Water', link: '/sde-coding/arrays/trapping-rain-water' },
            { text: 'Rotate Array', link: '/sde-coding/arrays/rotate-array' },
            { text: 'Move Zeros', link: '/sde-coding/arrays/move-zeros' },
            { text: 'Remove Duplicates', link: '/sde-coding/arrays/remove-duplicates' },
            { text: 'Find Peak', link: '/sde-coding/arrays/find-peak' },
            { text: 'Search Rotated', link: '/sde-coding/arrays/search-rotated' },
            { text: 'First Missing Positive', link: '/sde-coding/arrays/first-missing-positive' },
            { text: 'Median Two Arrays', link: '/sde-coding/arrays/median-two-arrays' },
            { text: 'Sliding Window Max', link: '/sde-coding/arrays/sliding-window-max' },
            { text: 'Subarray Sum K', link: '/sde-coding/arrays/subarray-sum-k' },
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
            { text: 'Two Sum', link: '/sde-coding/hash-tables/two-sum' },
            { text: 'Valid Anagram', link: '/sde-coding/hash-tables/valid-anagram' },
            { text: 'Group Anagrams', link: '/sde-coding/hash-tables/group-anagrams' },
            { text: 'Contains Duplicate', link: '/sde-coding/hash-tables/contains-duplicate' },
            { text: 'Top K Frequent', link: '/sde-coding/hash-tables/top-k-frequent' },
            { text: 'Longest Substring', link: '/sde-coding/hash-tables/longest-substring' },
            { text: 'Longest Consecutive', link: '/sde-coding/hash-tables/longest-consecutive' },
            { text: 'Subarray Sum K', link: '/sde-coding/hash-tables/subarray-sum-k' },
            { text: 'Minimum Window', link: '/sde-coding/hash-tables/minimum-window' },
            { text: 'LRU Cache', link: '/sde-coding/hash-tables/lru-cache' },
            { text: 'Design HashMap', link: '/sde-coding/hash-tables/design-hashmap' },
            { text: 'Design File System', link: '/sde-coding/hash-tables/design-file-system' },
            { text: 'All O One', link: '/sde-coding/hash-tables/all-oone' },
            { text: 'First Non Repeating', link: '/sde-coding/hash-tables/first-non-repeating' },
            { text: 'Four Sum II', link: '/sde-coding/hash-tables/four-sum-ii' },
            { text: 'Copy Random List', link: '/sde-coding/hash-tables/copy-random-list' },
            { text: 'Ransom Note', link: '/sde-coding/hash-tables/ransom-note' },
            { text: 'Max Profit', link: '/sde-coding/hash-tables/max-profit' },
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
            { text: 'Two Pointer', link: '/sde-coding/searching-sorting/two-pointer' },
            { text: 'Sliding Window', link: '/sde-coding/searching-sorting/sliding-window' },
            { text: 'Quick Select', link: '/sde-coding/searching-sorting/quick-select' },
            { text: 'Non-Comparison Sorts', link: '/sde-coding/searching-sorting/non-comparison-sorts' },
            { text: 'Merge Intervals', link: '/sde-coding/searching-sorting/merge-intervals' },
            { text: 'Rotated Array Search', link: '/sde-coding/searching-sorting/rotated-array-search' },
            { text: 'Rotated Arrays & Koko', link: '/sde-coding/searching-sorting/rotated-koko' },
            { text: 'K-Messed & Duplicates', link: '/sde-coding/searching-sorting/kmessed-duplicates' }
          ]
        },
        {
          text: 'Strings',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/strings/' },
            { text: 'Reverse String', link: '/sde-coding/strings/reverse-string' },
            { text: 'First Unique Char', link: '/sde-coding/strings/first-unique-char' },
            { text: 'String to Integer', link: '/sde-coding/strings/string-to-integer' },
            { text: 'Longest Common Prefix', link: '/sde-coding/strings/longest-common-prefix' },
            { text: 'Longest Palindromic Substring', link: '/sde-coding/strings/longest-palindromic-substring' },
            { text: 'Group Anagrams', link: '/sde-coding/strings/group-anagrams' },
            { text: 'Anagram Problems', link: '/sde-coding/strings/anagram-problems' },
            { text: 'Minimum Window Substring', link: '/sde-coding/strings/minimum-window-substring' },
            { text: 'Edit Distance', link: '/sde-coding/strings/edit-distance' },
            { text: 'Regex Matching', link: '/sde-coding/strings/regex-matching' },
            { text: 'Wildcard Matching', link: '/sde-coding/strings/wildcard-matching' },
            { text: 'Palindrome Partitioning', link: '/sde-coding/strings/palindrome-partitioning' },
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
            { text: 'Clone Graph', link: '/sde-coding/graphs/clone-graph' },
            { text: 'Connected Components', link: '/sde-coding/graphs/connected-components' },
            { text: 'Cycle Detection', link: '/sde-coding/graphs/cycle-detection' },
            { text: 'Graph Valid Tree', link: '/sde-coding/graphs/graph-valid-tree' },
            { text: 'Network Delay', link: '/sde-coding/graphs/network-delay' },
            { text: 'Cheapest Flights', link: '/sde-coding/graphs/cheapest-flights' },
            { text: 'Pacific Atlantic', link: '/sde-coding/graphs/pacific-atlantic' },
            { text: 'Surrounded Regions', link: '/sde-coding/graphs/surrounded-regions' },
            { text: 'Word Ladder', link: '/sde-coding/graphs/word-ladder' },
            { text: 'Alien Dictionary', link: '/sde-coding/graphs/alien-dictionary' },
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
            { text: 'BST Operations', link: '/sde-coding/trees/bst-operations' },
            { text: 'Construct & Serialize', link: '/sde-coding/trees/construct-serialize' },
            { text: 'Level Order & Zigzag', link: '/sde-coding/trees/level-order-zigzag' },
            { text: 'LCA & Path Sum', link: '/sde-coding/trees/lca-path-sum' },
            { text: 'Max Path Sum', link: '/sde-coding/trees/max-path-sum' },
            { text: 'Same Subtree & Invert', link: '/sde-coding/trees/same-subtree-invert' },
            { text: 'Good Nodes & Boundary', link: '/sde-coding/trees/good-nodes-boundary' },
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
            { text: 'Valid Parentheses', link: '/sde-coding/stacks-queues/valid-parentheses' },
            { text: 'Min Stack', link: '/sde-coding/stacks-queues/min-stack' },
            { text: 'Daily Temperatures', link: '/sde-coding/stacks-queues/daily-temperatures' },
            { text: 'Next Greater Element', link: '/sde-coding/stacks-queues/next-greater-element' },
            { text: 'Largest Rectangle Histogram', link: '/sde-coding/stacks-queues/largest-rectangle-histogram' },
            { text: 'Trapping Rain Water', link: '/sde-coding/stacks-queues/trapping-rain-water' },
            { text: 'Decode String', link: '/sde-coding/stacks-queues/decode-string' },
            { text: 'Simplify Path', link: '/sde-coding/stacks-queues/simplify-path' },
            { text: 'Asteroid Collision', link: '/sde-coding/stacks-queues/asteroid-collision' },
            { text: 'Basic Calculator', link: '/sde-coding/stacks-queues/basic-calculator' },
            { text: 'Basic Calculator II', link: '/sde-coding/stacks-queues/basic-calculator-ii' },
            { text: 'Reverse Polish Notation', link: '/sde-coding/stacks-queues/reverse-polish-notation' },
            { text: 'Remove K Digits', link: '/sde-coding/stacks-queues/remove-k-digits' },
            { text: 'Stock Span', link: '/sde-coding/stacks-queues/stock-span' },
            { text: 'Queue Using Stacks', link: '/sde-coding/stacks-queues/queue-using-stacks' },
            { text: 'Stack Using Queues', link: '/sde-coding/stacks-queues/stack-using-queues' },
            { text: 'Sliding Window Maximum', link: '/sde-coding/stacks-queues/sliding-window-maximum' },
            { text: 'Rotting Oranges', link: '/sde-coding/stacks-queues/rotting-oranges' },
            { text: 'Task Scheduler', link: '/sde-coding/stacks-queues/task-scheduler' },
            { text: 'First Negative Window', link: '/sde-coding/stacks-queues/first-negative-window' },
            { text: 'Parentheses & Min Stack', link: '/sde-coding/stacks-queues/parentheses-stack' },
            { text: 'Temperatures & Path', link: '/sde-coding/stacks-queues/temperatures-path' }
          ]
        },
        {
          text: 'Linked Lists',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/linked-lists/' },
            { text: 'Fast Slow Pointers', link: '/sde-coding/linked-lists/fast-slow-pointers' },
            { text: 'Merge Operations', link: '/sde-coding/linked-lists/merge-operations' },
            { text: 'Deletion Operations', link: '/sde-coding/linked-lists/deletion-operations' },
            { text: 'Reorder Operations', link: '/sde-coding/linked-lists/reorder-operations' },
            { text: 'Rotation Operations', link: '/sde-coding/linked-lists/rotation-operations' },
            { text: 'Swap Operations', link: '/sde-coding/linked-lists/swap-operations' },
            { text: 'Sort Operations', link: '/sde-coding/linked-lists/sort-operations' },
            { text: 'Intersection Operations', link: '/sde-coding/linked-lists/intersection-operations' },
            { text: 'Arithmetic Operations', link: '/sde-coding/linked-lists/arithmetic-operations' },
            { text: 'Advanced Operations', link: '/sde-coding/linked-lists/advanced-operations' },
            { text: 'Reverse K Group', link: '/sde-coding/linked-lists/reverse-k-group' },
            { text: 'LRU Cache', link: '/sde-coding/linked-lists/lru-cache' },
            { text: 'Reverse & Cycle', link: '/sde-coding/linked-lists/reverse-cycle' },
            { text: 'Merge & LRU Cache', link: '/sde-coding/linked-lists/merge-lru' }
          ]
        },
        {
          text: 'Heaps',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/heaps/' },
            { text: 'Kth Largest', link: '/sde-coding/heaps/kth-largest' },
            { text: 'Top K Frequent', link: '/sde-coding/heaps/top-k-frequent' },
            { text: 'K Closest Points', link: '/sde-coding/heaps/k-closest-points' },
            { text: 'Find Median Stream', link: '/sde-coding/heaps/find-median-stream' },
            { text: 'Merge K Sorted', link: '/sde-coding/heaps/merge-k-sorted' },
            { text: 'Meeting Rooms II', link: '/sde-coding/heaps/meeting-rooms-ii' },
            { text: 'Task Scheduler', link: '/sde-coding/heaps/task-scheduler' },
            { text: 'Reorganize String', link: '/sde-coding/heaps/reorganize-string' },
            { text: 'Smallest Range', link: '/sde-coding/heaps/smallest-range' },
            { text: 'Sliding Window Median', link: '/sde-coding/heaps/sliding-window-median' },
            { text: 'Heap Problems', link: '/sde-coding/heaps/heap-problems' }
          ]
        },
        {
          text: 'Recursion & Backtracking',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/sde-coding/recursion-backtracking/' },
            { text: 'Factorial', link: '/sde-coding/recursion-backtracking/factorial' },
            { text: 'Fibonacci', link: '/sde-coding/recursion-backtracking/fibonacci' },
            { text: 'Climbing Stairs', link: '/sde-coding/recursion-backtracking/climbing-stairs' },
            { text: 'Generate Parentheses', link: '/sde-coding/recursion-backtracking/generate-parentheses' },
            { text: 'Letter Combinations', link: '/sde-coding/recursion-backtracking/letter-combinations' },
            { text: 'Subsets', link: '/sde-coding/recursion-backtracking/subsets' },
            { text: 'Subsets II', link: '/sde-coding/recursion-backtracking/subsets-ii' },
            { text: 'Permutations', link: '/sde-coding/recursion-backtracking/permutations' },
            { text: 'Permutations II', link: '/sde-coding/recursion-backtracking/permutations-ii' },
            { text: 'Combination Sum', link: '/sde-coding/recursion-backtracking/combination-sum' },
            { text: 'Combination Sum II', link: '/sde-coding/recursion-backtracking/combination-sum-ii' },
            { text: 'Palindrome Partitioning', link: '/sde-coding/recursion-backtracking/palindrome-partitioning' },
            { text: 'Word Search', link: '/sde-coding/recursion-backtracking/word-search' },
            { text: 'Word Search II', link: '/sde-coding/recursion-backtracking/word-search-ii' },
            { text: 'N Queens', link: '/sde-coding/recursion-backtracking/n-queens' },
            { text: 'N Queens II', link: '/sde-coding/recursion-backtracking/n-queens-ii' },
            { text: 'Sudoku Solver', link: '/sde-coding/recursion-backtracking/sudoku-solver' },
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
            { text: 'Wildcard Matching', link: '/sde-coding/dynamic-programming/wildcard-matching' },
            { text: 'Climbing & Knapsack', link: '/sde-coding/dynamic-programming/climbing-knapsack' },
            { text: 'Coin Change & Paths', link: '/sde-coding/dynamic-programming/coin-change-paths' }
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
            { text: 'Hypothesis Testing', link: '/ml-fundamentals/interview-faq/statistics/hypothesis-testing' },
            { text: 'Overfitting & Underfitting', link: '/ml-fundamentals/interview-faq/overfitting-underfitting' },
            { text: 'Class Imbalance', link: '/ml-fundamentals/interview-faq/class-imbalance' },
            { text: 'Feature Engineering', link: '/ml-fundamentals/interview-faq/feature-engineering' },
            { text: 'Hyperparameter Tuning', link: '/ml-fundamentals/interview-faq/hyperparameter-tuning' },
            { text: 'Evaluation Metrics', link: '/ml-fundamentals/interview-faq/evaluation-metrics' },
            { text: 'Probability & Bayes', link: '/ml-fundamentals/interview-faq/probability-bayes' },
            { text: 'Vanishing Gradients', link: '/ml-fundamentals/interview-faq/vanishing-gradients' },
            { text: 'Transfer Learning', link: '/ml-fundamentals/interview-faq/transfer-learning' },
            { text: 'Model Drift', link: '/ml-fundamentals/interview-faq/model-drift' },
            { text: 'Interpretability (SHAP/LIME)', link: '/ml-fundamentals/interview-faq/interpretability' }
          ]
        }
      ],

      '/ml-coding/': [
        {
          text: 'Getting Started',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/ml-coding/' },
            { text: 'Approach & Patterns', link: '/ml-coding/coding-overview' },
            { text: 'Quick Reference', link: '/ml-coding/coding-reference' }
          ]
        },
        {
          text: 'Classical ML',
          collapsed: false,
          items: [
            { text: 'Linear Regression', link: '/ml-coding/linear-regression-implementation' },
            { text: 'Logistic Regression', link: '/ml-coding/logistic-regression-implementation' },
            { text: 'KNN', link: '/ml-coding/knn-implementation' },
            { text: 'K-Means', link: '/ml-coding/kmeans-implementation' },
            { text: 'Decision Tree', link: '/ml-coding/decision-tree-implementation' },
            { text: 'Random Forest', link: '/ml-coding/random-forest' },
            { text: 'Naive Bayes', link: '/ml-coding/naive-bayes' },
            { text: 'SVM', link: '/ml-coding/svm-implementation' },
            { text: 'PCA', link: '/ml-coding/pca-implementation' }
          ]
        },
        {
          text: 'Deep Learning',
          collapsed: false,
          items: [
            { text: 'Neural Network MLP', link: '/ml-coding/neural-network-mlp' },
            { text: 'Softmax & Cross-Entropy', link: '/ml-coding/softmax-crossentropy' },
            { text: 'Batch Norm & Dropout', link: '/ml-coding/batch-norm-dropout' },
            { text: 'Self-Attention', link: '/ml-coding/self-attention' },
            { text: '2D Convolution', link: '/ml-coding/cnn-filter' },
            { text: 'Feature Scaling', link: '/ml-coding/feature-scaling' }
          ]
        },
        {
          text: 'Practice',
          collapsed: true,
          items: [
            { text: 'Practical Problems', link: '/ml-coding/practical-problems' }
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
          text: 'Core Framework',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/behavioural/' },
            { text: 'STAR Framework', link: '/behavioural/star-framework' },
            { text: 'Quick Reference', link: '/behavioural/quick-reference' }
          ]
        },
        {
          text: 'Company Guides',
          collapsed: false,
          items: [
            { text: 'Google', link: '/behavioural/companies/google' },
            { text: 'Amazon', link: '/behavioural/companies/amazon' },
            { text: 'Meta', link: '/behavioural/companies/meta' },
            { text: 'Microsoft', link: '/behavioural/companies/microsoft' },
            { text: 'AI Labs', link: '/behavioural/companies/ai-labs' },
            { text: 'Apple, Netflix, Stripe', link: '/behavioural/companies/apple-netflix-stripe' }
          ]
        },
        {
          text: 'Practice Resources',
          collapsed: false,
          items: [
            { text: 'Common Questions', link: '/behavioural/common-questions' },
            { text: 'Questions Bank (148)', link: '/behavioural/questions-bank' },
            { text: 'Story Templates', link: '/behavioural/story-templates' }
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

    socialLinks: [],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Developed by <a href="https://abhinaavramesh.github.io/portfolio/" target="_blank">Abhinaav Ramesh</a>',
      copyright: 'The Handbook — Technical Interview Preparation'
    }
  },

  markdown: {
    lineNumbers: true,
    math: true,
    config: (md) => {
      md.use(mathjax3)
      md.use(taskLists, { enabled: true })
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
    theme: 'base',
    themeVariables: {
      primaryColor: '#2563eb',
      primaryTextColor: '#1e293b',
      primaryBorderColor: '#3b82f6',
      lineColor: '#3b82f6',
      secondaryColor: '#f1f5f9',
      tertiaryColor: '#e2e8f0',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      fontSize: '14px',
      nodeBorder: '#3b82f6',
      clusterBkg: '#f8fafc',
      clusterBorder: '#e2e8f0',
      titleColor: '#1e293b',
      edgeLabelBackground: '#ffffff'
    },
    flowchart: {
      htmlLabels: true,
      curve: 'basis',
      nodeSpacing: 50,
      rankSpacing: 50,
      padding: 15,
      useMaxWidth: true
    },
    mindmap: {
      padding: 20,
      useMaxWidth: true
    }
  },
  mermaidPlugin: {
    class: 'mermaid'
  }
}))
