import { test, expect } from '@playwright/test';

test('citation links resolve to live pages', async ({ page }) => {
  test.slow();

  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });

  const query =
    'Vi trenger et nytt modulbasert kurs som dekker AI-etikk, dataminimering og ansvarlig bruk av maskinlæring i helsesektoren.';
  await page.fill('#queryInput', query);
  await page.click('#searchBtn');

  const firstResult = page.locator('.result-card').first();
  await expect(firstResult).toBeVisible({ timeout: 120_000 });

  await firstResult.locator('[data-action="open-details"]').click();
  const links = page.locator('#modalBody .modal-citation a');
  await expect(links.first()).toBeVisible({ timeout: 10_000 });

  const hrefs = await links.evaluateAll((nodes) => nodes.map((n) => (n as HTMLAnchorElement).href));
  expect(hrefs.length).toBeGreaterThan(0);

  for (const href of hrefs) {
    const resp = await page.request.fetch(href, { method: 'GET', maxRedirects: 5, timeout: 15_000 });
    expect(resp.status(), `URL failed: ${href}`).toBeLessThan(400);
  }
});
