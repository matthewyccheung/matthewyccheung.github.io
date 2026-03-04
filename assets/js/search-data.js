// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/CV/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "Selected publications and research highlights.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "post-risk-averse-calibration-rac-from-risk-averse-decisions-to-optimal-prediction-sets",
        
          title: "Risk-Averse Calibration (RAC): From Risk-Averse Decisions to Optimal Prediction Sets",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rac/";
          
        },
      },{id: "news-wearing-a-mask-compressed-representations-of-variable-length-sequences-using-recurrent-neural-tangent-kernels-was-accepted-at-icassp-2021",
          title: 'Wearing a MASK: Compressed Representations of Variable-Length Sequences Using Recurrent Neural Tangent Kernels...',
          description: "",
          section: "News",},{id: "news-appointed-to-the-national-library-of-medicine-nlm-training-program-in-biomedical-informatics-and-data-science",
          title: 'Appointed to the National Library of Medicine (NLM) Training Program in Biomedical Informatics...',
          description: "",
          section: "News",},{id: "news-defended-m-s-thesis-on-wearable-blood-pressure-monitoring-and-study-design",
          title: 'Defended M.S. thesis on Wearable Blood Pressure Monitoring and Study Design.',
          description: "",
          section: "News",},{id: "news-wearablebp-github-io-is-live-this-website-accompanies-our-review-at-ieee-transactions-on-biomedical-engineering",
          title: 'wearablebp.github.io is live. This website accompanies our review at IEEE Transactions on Biomedical...',
          description: "",
          section: "News",},{id: "news-collaborative-work-on-developing-an-ultrasound-based-non-invasive-intracranial-pressure-monitoring-device-17-11-a-9mw-ultrasonic-through-transmission-transceiver-for-non-invasive-intracranial-pressure-sensing-was-accepted-at-ieee-international-solid-state-circuits-conference",
          title: 'Collaborative work on developing an ultrasound-based non-invasive intracranial pressure monitoring device 17.11 A...',
          description: "",
          section: "News",},{id: "news-after-several-years-of-dedicated-work-our-paper-wearable-blood-pressure-monitoring-devices-understanding-heterogeneity-in-design-and-evaluation-is-accepted-and-available-on-ieee-transactions-on-biomedical-engineering",
          title: 'After several years of dedicated work, our paper Wearable Blood Pressure Monitoring Devices:...',
          description: "",
          section: "News",},{id: "news-collaborative-follow-up-work-on-developing-an-ultrasound-based-non-invasive-intracranial-pressure-monitoring-device-an-ultrasonic-transceiver-for-non-invasive-intracranial-pressure-sensing-was-accepted-at-ieee-transactions-on-biomedical-circuits-and-systems",
          title: 'Collaborative follow-up work on developing an ultrasound-based non-invasive intracranial pressure monitoring device An...',
          description: "",
          section: "News",},{id: "news-our-paper-when-are-diffusion-priors-helpful-in-sparse-reconstruction-a-study-with-sparse-view-ct-was-accepted-at-ieee-isbi-2025",
          title: 'Our paper When are Diffusion Priors Helpful in Sparse Reconstruction? A Study With...',
          description: "",
          section: "News",},{id: "news-our-paper-metric-guided-conformal-bounds-for-probabilistic-image-reconstruction-was-accepted-at-unsure-miccai-as-a-long-oral-presentation",
          title: 'Our paper Metric-guided Conformal Bounds for Probabilistic Image Reconstruction was accepted at UNSURE...',
          description: "",
          section: "News",},{id: "news-our-paper-bias-aware-conformal-prediction-for-metric-based-imaging-pipelines-was-accepted-at-ieee-isbi-2026",
          title: 'Our paper Bias-Aware Conformal Prediction for Metric-Based Imaging Pipelines was accepted at IEEE...',
          description: "",
          section: "News",},{id: "news-our-paper-compass-robust-feature-conformal-prediction-for-medical-segmentation-metrics-was-accepted-at-iclr-2026",
          title: 'Our paper COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics was accepted...',
          description: "",
          section: "News",},{id: "news-new-paper-on-arxiv-efficient-conformal-volumetry-for-template-based-segmentation",
          title: 'New paper on arXiv Efficient Conformal Volumetry for Template-Based Segmentation.',
          description: "",
          section: "News",},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6D%61%74%74%79%63@%72%69%63%65.%65%64%75", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=6d3hfUcAAAAJ", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/matthewyccheung", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/matthewyccheung", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/matte_ce", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
