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

  test('user can search and see matches', async ({ page }, testInfo) => {
    test.slow();
    const consoleMessages: string[] = [];
    page.on('console', (msg) => {
      consoleMessages.push(`[console:${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', (error) => {
      consoleMessages.push(`[pageerror] ${error.message}`);
    });

    await test.step('load home view and submit search', async () => {
      await page.goto('/');
      await expect(page.locator('h1')).toContainText('ONH Expert Connector');
      const queryInput = page.locator('#queryInput');
      const searchBtn = page.locator('#searchBtn');

      await queryInput.fill(
        'Vi trenger et nytt modulbasert kurs som dekker AI-etikk, dataminimering og ansvarlig bruk av maskinlæring i helsesektoren.'
      );
      await expect(searchBtn).toBeEnabled();
      await searchBtn.click();
    });

    await test.step('view results and capture artifacts', async () => {
      const firstResult = page.locator('.result-card').first();
      await expect(firstResult).toBeVisible({ timeout: 180_000 });

      const statusLine = (await page.locator('#statusLine').textContent())?.trim();
      consoleMessages.push(`[ui] statusLine=${statusLine}`);

      // open details modal for first result to ensure it renders
      await firstResult.locator('[data-action=\"open-details\"]').click();
      await expect(page.locator('#modalBody')).toBeVisible({ timeout: 10_000 });
      await page.click('#modalCloseBtn');

      const screenshotPath = path.join(artifactsDir, 'results.png');
      await page.screenshot({ path: screenshotPath, fullPage: true });
    });

    const logPath = path.join(artifactsDir, `playwright-console-${testInfo.retry + 1}.log`);
    writeFileSync(logPath, consoleMessages.join('\n'), 'utf-8');
    await testInfo.attach('console-log', {
      body: consoleMessages.join('\n'),
      contentType: 'text/plain',
    });
  });
});
