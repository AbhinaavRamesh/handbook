import fs from 'fs'
import path from 'path'
import type { DefaultTheme, SiteConfig } from 'vitepress'

const SECTION_TITLES: Record<string, string> = {
  '/sde-coding/': 'SDE Coding',
  '/ml-fundamentals/': 'ML Fundamentals',
  '/genai/': 'GenAI Engineering',
  '/ml-coding/': 'ML Coding (From Scratch)',
  '/ml-design/': 'ML System Design',
  '/behavioural/': 'Behavioural Interviews'
}

interface PageRef {
  text: string
  link: string
}

// Personal content is local-only and must never be published
function collectLinks(items: DefaultTheme.SidebarItem[] | undefined, out: PageRef[]) {
  for (const item of items ?? []) {
    if (item.link && item.text && !item.link.includes('/personal/')) {
      out.push({ text: item.text, link: item.link })
    }
    if (item.items) collectLinks(item.items, out)
  }
}

function linkToFile(srcDir: string, link: string): string {
  const rel = link.endsWith('/') ? `${link}index.md` : `${link}.md`
  return path.join(srcDir, rel)
}

function linkToUrl(siteUrl: string, link: string): string {
  const clean = link.replace(/^\//, '')
  return link.endsWith('/') ? siteUrl + clean : `${siteUrl}${clean}.html`
}

function stripFrontmatter(src: string): string {
  return src.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '')
}

/**
 * Emits llms.txt (curated index, mirrors the sidebar) and llms-full.txt
 * (full concatenated content) into the build output directory, following
 * the https://llmstxt.org/ convention.
 */
export function generateLlmsArtifacts(siteConfig: SiteConfig, siteUrl: string) {
  const { srcDir, outDir, site } = siteConfig
  const sidebar = (site.themeConfig as DefaultTheme.Config).sidebar as DefaultTheme.SidebarMulti

  const sectionKeys = [
    ...Object.keys(SECTION_TITLES).filter((key) => key in sidebar),
    ...Object.keys(sidebar).filter((key) => !(key in SECTION_TITLES))
  ]

  const header = [`# ${site.title}`, '', `> ${site.description}`, '']
  const index = [...header]
  const full = [...header]

  for (const key of sectionKeys) {
    const pages: PageRef[] = []
    collectLinks(sidebar[key] as DefaultTheme.SidebarItem[], pages)

    index.push(`## ${SECTION_TITLES[key] ?? key}`, '')
    const seen = new Set<string>()
    for (const page of pages) {
      if (seen.has(page.link)) continue
      seen.add(page.link)

      const file = linkToFile(srcDir, page.link)
      if (!fs.existsSync(file)) continue

      const url = linkToUrl(siteUrl, page.link)
      index.push(`- [${page.text}](${url})`)

      const body = stripFrontmatter(fs.readFileSync(file, 'utf-8')).trim()
      full.push('---', '', `URL: ${url}`, '', body, '')
    }
    index.push('')
  }

  fs.writeFileSync(path.join(outDir, 'llms.txt'), index.join('\n'))
  fs.writeFileSync(path.join(outDir, 'llms-full.txt'), full.join('\n'))
}
