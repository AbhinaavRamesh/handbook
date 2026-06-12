<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import StudyProgress from './components/StudyProgress.vue'
</script>

<template>
  <DefaultTheme.Layout>
    <template #doc-before>
      <StudyProgress />
    </template>
  </DefaultTheme.Layout>
</template>
