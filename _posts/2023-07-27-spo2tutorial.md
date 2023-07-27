---
layout: single
classes: wide
author_profile: true
title: "[Paper Summary] A review of the effect of skin pigmentation on pulse oximeter accuracy"
show_date: true
comments: true
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

<script type="text/javascript"
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>


[Paper Link, DOI: 10.1088/1361-6579/acd51a](https://iopscience.iop.org/article/10.1088/1361-6579/acd51a/meta)

Summary: Review presents an introduction to the technique of pulse oximetry including its basic principle of operation, technology, and limitations, with a focus on skin pigmentation.

Importance: PulseOxs have been known for a while now to perform worse on people with darker skin. Understanding why this is the case is important for fairness. This is manifested in the assumptions we make in the derivation.

<h2> SpO2 derivation </h2>

<h4> Modified Beer Lambert </h4>

Light absorption through skin can be modeled using the modified beer-lambert law.

$$I_t = I_0 e^{-G-\epsilon C l}$$

$$G$$: light loss due to scattering

$$d$$ or $$l$$: pathlength or optical path length (actual distance traveled by light), which is longer than the path length due to scattering

$$I_t$$: intensity of transmitted light

$$I_0$$: intensity of incident light

$$\epsilon$$: molar extinction concentration of tissue layers and chromophores such as hemoglobin, melanin, water, etc.

$$C$$: concentrations of tissue layers and chromophores

<h4> Find change in concentration:</h4>

$$\frac{I_D - I_S}{I_S} = \epsilon \Delta C l$$

$$I_D$$: maximum absorption in PPG 

$$I_S$$: minimum absorption in PPG

$$\Delta C$$: maximal change in hemoglobin in cardiac cycle

$$ln(\frac{I_t}{I_0}) = -G-\epsilon C l$$

$$ln(\frac{I_{max}}{I_0}) - ln(\frac{I_{min}}{I_0}) = -G_{max} - \epsilon C_{max} l + G_{min} + \epsilon C_{min} l$$

$$ln(\frac{I_{max}}{l_{min}}) = -\Delta G - \epsilon \Delta C l$$


$$R = \frac{ln(I_{max}/I_{min})_{660}}{ln(I_{max}/I_{min})_{940}} = \frac{[-\Delta G - \epsilon \Delta C l]_{660}}{[-\Delta G - \epsilon \Delta C l]_{940}}$$

<h4> Assume $\Delta G = 0$ over cardiac cycle </h4>

$$R = \frac{[\epsilon \Delta C l]_{660}}{[\epsilon \Delta C l]_{940}}$$

<h4> Assume $\Delta C$ is the same for 660 and 940 </h4>

$$R = \frac{[\epsilon l]_{660}}{[\epsilon l]_{940}}$$

<h4> Use $\epsilon = \epsilon_O SaO_2 + \epsilon_D(1-SaO_2)$ </h4>

$$\epsilon_O$$: extinction coefficients for oxygenated

$$\epsilon_D$$: deoxygenated blood

$$SpO_2 = \frac{(\epsilon_{D,660}-R(l_{940}/l_{660}) \epsilon_{D,940})}{(R(l_{940}/l_{660})(\epsilon_{O,940} - \epsilon_{D,940}) + (\epsilon_{D,660} - \epsilon_{O,660}))}$$

<h2> Major Assumptions and Compromises </h2>

<h4> Cardiac cycle assumptions: </h4>
1. Constant $$G$$
2. Constant $$I$$
3. Constant $$\epsilon$$
4. Contact $$C$$

<h4> Generalization assumptions: </h4>
1. $$\Delta C$$ approximately equal at two wavelengths
2. $$\frac{l_{940}}{l_{660}}$$ remains constant across individuals

Type of scattering (Rayleigh or Mie) is determined by the size of scatterer relative to the selected wavelength - found to be larger for individuals with darker skin. Photons tend to travel in a more forward direction (Mie scattering) in individuals with dark skin.

<h4> Compromises: </h4>
1. Ground truth measures functional hemoglobin (includes carboxyhemoglobin (COHb) and methaemoglobin (MetHb). Can be influenced by high concentrations or partial pressures of carbon monoxide and/or other gases and chemical compounds. In visible spectrum, COHb has similar absorption profile to oxyhemoglobin - cannot distinguish. But effect size is small.
2. Poor blood perfusion: pulse ox relies on optical measurements of arterial pulsations within tissue bed. But pulsations only make up 2-5% of total optical absorption profile and is dependent on adequate blood flow to measurement area.
3. Absorption corrupted by nail polish or artificial fingernails.
4. Ambient light
5. Intradermal dyes to identify anatomical structure (i.e. contrast imaging)
6. Traditional calibration method is limited in range due to ethics
