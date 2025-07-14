<template>
  <div class="emulation-report bg-gray-900 text-white p-6 rounded-lg shadow-md">
    <h2 class="text-2xl font-bold mb-4">Emulation Report: {{ report.session_id }}</h2>

    <div class="grid grid-cols-2 gap-4 text-sm mb-6">
      <div><strong>Start Time:</strong> {{ formatDate(report.start_time) }}</div>
      <div><strong>End Time:</strong> {{ formatDate(report.end_time) }}</div>
      <div><strong>Status:</strong> <span :class="statusColor(report.status)">{{ report.status }}</span></div>
      <div><strong>Total Steps:</strong> {{ report.steps.length }}</div>
    </div>

    <div v-if="report.steps.length === 0" class="text-yellow-400">
      No step data found for this session.
    </div>

    <table v-else class="table-auto w-full text-sm border border-gray-700">
      <thead class="bg-gray-800">
        <tr>
          <th class="px-4 py-2 border-b border-gray-700 text-left">Step</th>
          <th class="px-4 py-2 border-b border-gray-700 text-left">Status</th>
          <th class="px-4 py-2 border-b border-gray-700 text-left">Output</th>
          <th class="px-4 py-2 border-b border-gray-700 text-left">Duration</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="step in report.steps" :key="step.name" class="hover:bg-gray-800">
          <td class="px-4 py-2 border-b border-gray-700">{{ step.name }}</td>
          <td class="px-4 py-2 border-b border-gray-700">
            <span :class="statusColor(step.status)">
              {{ step.status }}
            </span>
          </td>
          <td class="px-4 py-2 border-b border-gray-700">
            <code class="text-green-300">{{ stringifyOutput(step.output) }}</code>
          </td>
          <td class="px-4 py-2 border-b border-gray-700">
            {{ step.duration }}s
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script lang="ts" setup>
import { defineProps } from 'vue';

interface StepResult {
  name: string;
  status: 'success' | 'failed' | 'skipped';
  output: Record<string, any>;
  duration: number;
}

interface EmulationReport {
  session_id: string;
  start_time: string;
  end_time: string;
  status: 'complete' | 'failed' | 'partial';
  steps: StepResult[];
}

const props = defineProps<{
  report: EmulationReport;
}>();

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleString();
}

function stringifyOutput(output: Record<string, any>): string {
  try {
    return JSON.stringify(output, null, 2);
  } catch {
    return '[Unrenderable Output]';
  }
}

function statusColor(status: string): string {
  switch (status) {
    case 'success':
    case 'complete':
      return 'text-green-400';
    case 'failed':
      return 'text-red-400';
    case 'partial':
    case 'skipped':
      return 'text-yellow-400';
    default:
      return 'text-gray-400';
  }
}
</script>

<style scoped>
code {
  font-size: 0.85rem;
  word-break: break-word;
}
</style>
