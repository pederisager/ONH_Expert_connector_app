import { expect, test } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';

const artifactsDir = path.join('outputs', 'playwright');

function ensureArtifactsDir() {
  mkdirSync(artifactsDir, { recursive: true });
}

test.describe('ONH Expert Connector UI smoke flow', () => {
  test.beforeEach(() => {
    ensureArtifactsDir();
  });

  test('user can analyze a topic and fetch matches', async ({ page }, testInfo) => {
    test.slow();
    const consoleMessages: string[] = [];
    page.on('console', (msg) => {
      consoleMessages.push(`[console:${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', (error) => {
      consoleMessages.push(`[pageerror] ${error.message}`);
    });

    await test.step('load home view and submit analysis', async () => {
      await page.goto('/');
      await expect(page.locator('h1')).toContainText('ONH Expert Connector');
      const topicInput = page.locator('#topicInput');
      const analyzeBtn = page.locator('#analyzeBtn');
      await expect(analyzeBtn).toBeDisabled();

      await topicInput.fill(
        'Vi trenger et nytt modulbasert kurs som dekker AI-etikk, dataminimering og ansvarlig bruk av maskinlæring i helsesektoren.'
      );
      await expect(analyzeBtn).toBeEnabled();
      await analyzeBtn.click();

      const themesView = page.locator('#themesView');
      await expect(themesView).toHaveClass(/active/, { timeout: 60_000 });
      await expect(page.locator('#themesChips .chip').first()).toBeVisible({ timeout: 60_000 });
    });

    await test.step('run matching and capture UI artifacts', async () => {
      await page.click('#runMatchBtn');
      const firstResult = page.locator('.result-card').first();
      await expect(firstResult).toBeVisible({ timeout: 180_000 });

      const resultsCountText = (await page.locator('#resultsCount').textContent())?.trim();
      consoleMessages.push(`[ui] resultsCount=${resultsCountText}`);

      const screenshotPath = path.join(artifactsDir, 'results.png');
      await page.screenshot({ path: screenshotPath, fullPage: true });
    });

    const logPath = path.join(
      artifactsDir,
      `playwright-console-${testInfo.retry + 1}.log`
    );
    writeFileSync(logPath, consoleMessages.join('\n'), 'utf-8');
    await testInfo.attach('console-log', {
      body: consoleMessages.join('\n'),
      contentType: 'text/plain',
    });
  });
});
