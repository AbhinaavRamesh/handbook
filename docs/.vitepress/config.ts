import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import { withMermaid } from 'vitepress-plugin-mermaid'
import fs from 'fs'
import path from 'path'

// Check if personal content exists (local dev only)
const hasPersonalContent = fs.existsSync(path.resolve(__dirname, '../behavioural/personal'))

export default withMermaid(defineConfig({
  title: "ML Interview Playbook",
  description: "Complete preparation guide for SDE-ML, MLE, and AI Engineering roles",

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
          { text: 'Interview FAQ', link: '/ml-fundamentals/interview-faq/' }
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
          text: 'ML Fundamentals',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/ml-fundamentals/' },
            { text: 'How to Answer', link: '/ml-fundamentals/concepts-overview' },
            { text: 'Linear Regression', link: '/ml-fundamentals/linear-regression' },
            { text: 'Logistic Regression', link: '/ml-fundamentals/logistic-regression' },
            { text: 'Decision Trees', link: '/ml-fundamentals/decision-trees' },
            { text: 'SVM', link: '/ml-fundamentals/svm' },
            { text: 'KNN', link: '/ml-fundamentals/knn' },
            { text: 'Neural Networks', link: '/ml-fundamentals/neural-networks' },
            { text: 'Clustering', link: '/ml-fundamentals/clustering' },
            { text: 'Model Evaluation', link: '/ml-fundamentals/model-evaluation' }
          ]
        },
        {
          text: '📋 Interview FAQ',
          collapsed: false,
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
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'ML Interview Playbook',
      copyright: 'Your complete guide to SDE-ML interviews'
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
