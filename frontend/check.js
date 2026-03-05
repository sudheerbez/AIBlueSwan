import puppeteer from 'puppeteer';

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    page.on('console', msg => console.log('BROWSER_LOG:', msg.text()));
    page.on('pageerror', err => console.log('BROWSER_ERROR:', err.toString()));

    try {
        await page.goto('http://localhost:5173', { waitUntil: 'networkidle0', timeout: 15000 });
        console.log('Navigated properly.');
        const content = await page.content();
        console.log(content.substring(0, 500));
    } catch (err) {
        console.error('Failed to load:', err);
    }

    await browser.close();
})();
