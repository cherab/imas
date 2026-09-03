"use strict";

(function () {
    const root = document.querySelector("[data-md-component='outdated']");
    const banner = root?.querySelector(".version-banner");
    if (!root || !banner) return;

    const prefix = banner.querySelector("[data-version-banner-prefix]");
    const emphasis = banner.querySelector("[data-version-banner-emphasis]");
    const link = banner.querySelector("[data-version-banner-link]");
    const suffix = banner.querySelector("[data-version-banner-suffix]");
    const kind = banner.dataset.versionKind;
    const currentPath = banner.dataset.versionPath?.replace(/^\/+|\/+$/g, "");
    const manifestUrl = banner.dataset.versionManifest;
    const latestUrl = banner.dataset.latestUrl;

    document.documentElement.dataset.docsVersionKind = kind;

    function setVisibility(visible) {
        root.dataset.versionBannerVisible = String(visible);
        root.hidden = !visible;
    }

    function showVersionWarning(emphasisText) {
        prefix.textContent = "This is documentation for ";
        emphasis.textContent = emphasisText;
        suffix.textContent = ".";
        if (latestUrl) {
            link.textContent = "Switch to latest version";
            link.href = latestUrl;
            link.target = "";
            link.rel = "";
            link.hidden = false;
        } else {
            link.hidden = true;
        }
        setVisibility(true);
    }

    function pathFromVersion(version) {
        try {
            const siteRoot = new URL(".", manifestUrl);
            return new URL(version, siteRoot).pathname.replace(
                /^\/+|\/+$/g,
                "",
            );
        } catch {
            return "";
        }
    }

    function isCurrentPath(version) {
        const path = pathFromVersion(version);
        return path === currentPath || path.endsWith(`/${currentPath}`);
    }

    async function loadVersions() {
        if (!manifestUrl) return [];
        const response = await fetch(manifestUrl, { cache: "no-cache" });
        if (!response.ok) {
            throw new Error(`Unable to load version manifest (${response.status})`);
        }
        return response.json();
    }

    async function updateBanner() {
        if (!["development", "release", "pull-request"].includes(kind)) {
            setVisibility(false);
            return;
        }

        if (kind === "pull-request") {
            setVisibility(true);
            return;
        }

        if (kind === "development") {
            showVersionWarning("an unstable development version");
            return;
        }

        let versions = [];
        try {
            versions = await loadVersions();
        } catch (error) {
            // A release stays hidden because its age cannot be determined.
            console.warn(error);
        }

        const latestRelease = versions.find((entry) => entry.aliases?.includes("latest"));

        if (!latestRelease) {
            setVisibility(false);
            return;
        }

        if (isCurrentPath(latestRelease.version)) {
            setVisibility(false);
            return;
        }

        const version = currentPath?.split("/").at(-1)?.replace(/^v(?=\d)/, "") || "unknown";
        showVersionWarning(`an old version (${version})`);
    }

    // The theme's own outdated-version script runs while the document is being
    // parsed.  Updating after load ensures this more specific decision wins.
    window.addEventListener("load", () => {
        window.setTimeout(updateBanner, 0);
    });
})();
