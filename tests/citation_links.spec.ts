import { test, expect } from '@playwright/test';

test('citation links resolve to live pages', async ({ page }) => {
  test.slow();

  // Navigate to running app
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });

  // Fill topic and analyze
  const topic =
    'Vi trenger et nytt modulbasert kurs som dekker AI-etikk, dataminimering og ansvarlig bruk av maskinlæring i helsesektoren.';
  await page.fill('#topicInput', topic);
  await page.click('#analyzeBtn');

  // Wait for themes to appear
  await expect(page.locator('#themesChips .chip').first()).toBeVisible({ timeout: 60_000 });

  // Run match
  await page.click('#runMatchBtn');
  const firstResult = page.locator('.result-card').first();
  await expect(firstResult).toBeVisible({ timeout: 120_000 });

  // Open sources modal on first result
  await firstResult.locator('button[data-action="view-sources"]').click();
  const modal = page.locator('.modal-citation a');
  await expect(modal.first()).toBeVisible({ timeout: 10_000 });

  const hrefs = await modal.evaluateAll((nodes) => nodes.map((n) => (n as HTMLAnchorElement).href));
  expect(hrefs.length).toBeGreaterThan(0);

  // Validate each link responds with status < 400 (follow redirects)
  for (const href of hrefs) {
    const resp = await page.request.fetch(href, { method: 'GET', maxRedirects: 5, timeout: 15_000 });
    expect(resp.status(), `URL failed: ${href}`).toBeLessThan(400);
  }
});
