import anywidget
import traitlets

class StripeSankeyInline(anywidget.AnyWidget):
    _esm = """
    import * as d3 from "https://esm.sh/d3@7";

    function render({ model, el }) {
        el.innerHTML = '';

        const data = model.get("sankey_data");
        const width = model.get("width");
        const height = model.get("height");
        const colorSchemes = model.get("color_schemes");
        const selectedFlow = model.get("selected_flow");
        const metricMode = model.get("metric_mode");
        const metricConfig = model.get("metric_config");

        if (!data || !data.nodes || Object.keys(data.nodes).length === 0) {
            el.innerHTML = '<div style="padding: 20px; text-align: center; font-family: sans-serif;">No data available. Please load your processed data first.</div>';
            return;
        }

        // Create SVG
        const svg = d3.select(el)
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .style("background", "#fafafa")
            .style("border", "1px solid #ddd");

        // Adjust top margin based on color mode - 'metrics' mode needs more space for perplexity bars
        const topMargin = model.get("color_mode") === 'metrics' ? 120 : 60;
        const margin = { top: topMargin, right: 150, bottom: 60, left: 100 }; // Increased right margin for tooltips
        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left}, ${margin.top})`);

        // Process data for visualization
        const processedData = processDataForVisualization(data);

        // Calculate metric scales if in metric mode
        let metricScales = null;
        if (metricMode) {
            metricScales = calculateMetricScales(processedData, data, metricConfig);
        }

        // Draw the actual sankey diagram
        drawSankeyDiagram(g, processedData, chartWidth, chartHeight, colorSchemes, selectedFlow, model, metricMode, metricScales, metricConfig);

        // No metric legend - removed to avoid clutter

        // Update on data change
        model.on("change:sankey_data", () => {
            const newData = model.get("sankey_data");
            if (newData && Object.keys(newData).length > 0) {
                const newProcessedData = processDataForVisualization(newData);
                let newMetricScales = null;
                if (model.get("metric_mode")) {
                    newMetricScales = calculateMetricScales(newProcessedData, newData, model.get("metric_config"));
                }
                svg.selectAll("*").remove();
                const newG = svg.append("g").attr("transform", `translate(${margin.left}, ${margin.top})`);
                drawSankeyDiagram(newG, newProcessedData, chartWidth, chartHeight, colorSchemes, model.get("selected_flow"), model, model.get("metric_mode"), newMetricScales, model.get("metric_config"));
                // No metric legend - removed
            }
        });

        // Update on metric mode change
        model.on("change:metric_mode", () => {
            const newMetricMode = model.get("metric_mode");
            let newMetricScales = null;
            if (newMetricMode) {
                newMetricScales = calculateMetricScales(processedData, data, model.get("metric_config"));
            }
            
            // Recalculate margin based on new color mode
            const newTopMargin = model.get("color_mode") === 'metrics' ? 120 : 60;
            const newMargin = { top: newTopMargin, right: 150, bottom: 60, left: 100 };
            const newChartWidth = width - newMargin.left - newMargin.right;
            const newChartHeight = height - newMargin.top - newMargin.bottom;
            
            svg.selectAll("*").remove();
            const newG = svg.append("g").attr("transform", `translate(${newMargin.left}, ${newMargin.top})`);
            drawSankeyDiagram(newG, processedData, newChartWidth, newChartHeight, colorSchemes, model.get("selected_flow"), model, newMetricMode, newMetricScales, model.get("metric_config"));
            // No metric legend - removed
        });

        // Update on selected flow change
        model.on("change:selected_flow", () => {
            const newSelectedFlow = model.get("selected_flow");
            svg.selectAll("*").remove();
            const newG = svg.append("g").attr("transform", `translate(${margin.left}, ${margin.top})`);
            let newMetricScales = null;
            if (model.get("metric_mode")) {
                newMetricScales = calculateMetricScales(processedData, data, model.get("metric_config"));
            }
            drawSankeyDiagram(newG, processedData, chartWidth, chartHeight, colorSchemes, newSelectedFlow, model, model.get("metric_mode"), newMetricScales, model.get("metric_config"));
            // No metric legend - removed
        });
    }

    function calculateMetricScales(processedData, rawData, metricConfig) {
        console.log("Calculating metric scales...");

        const perplexityValues = [];
        const coherenceValues = [];

        // Extract metric values from all nodes
        processedData.nodes.forEach(node => {
            const nodeData = rawData.nodes[node.id];
            if (nodeData) {
                // Get perplexity from model_metrics
                if (nodeData.model_metrics && nodeData.model_metrics.perplexity !== undefined) {
                    perplexityValues.push(nodeData.model_metrics.perplexity);
                }

                // Get coherence from mallet_diagnostics
                if (nodeData.mallet_diagnostics && nodeData.mallet_diagnostics.coherence !== undefined) {
                    coherenceValues.push(nodeData.mallet_diagnostics.coherence);
                }
            }
        });

        console.log(`Found ${perplexityValues.length} perplexity values, ${coherenceValues.length} coherence values`);

        if (perplexityValues.length === 0 || coherenceValues.length === 0) {
            console.warn("Insufficient metric data for metric mode");
            return null;
        }

        // Create scales
        const perplexityExtent = d3.extent(perplexityValues);
        const coherenceExtent = d3.extent(coherenceValues);

        console.log("Perplexity range:", perplexityExtent);
        console.log("Coherence range:", coherenceExtent);

        // Perplexity: lower is better, so we invert the scale (low perplexity = high red intensity)
        const perplexityScale = d3.scaleLinear()
            .domain(perplexityExtent)
            .range([1, 0]); // Inverted: low perplexity gets high value (more red)

        // Coherence: higher is better (less negative), but values are negative
        // More negative = worse, less negative = better
        const coherenceScale = d3.scaleLinear()
            .domain(coherenceExtent)
            .range([0, 1]); // Less negative coherence gets high value (more blue)

        return {
            perplexity: perplexityScale,
            coherence: coherenceScale,
            perplexityExtent,
            coherenceExtent
        };
    }

    function getMetricColor(nodeId, rawData, metricScales, metricConfig, colorMode) {
        if (!metricScales) return "#666";

        const nodeData = rawData.nodes[nodeId];
        if (!nodeData) return "#666";

        let perplexityValue = null;
        let coherenceValue = null;

        // Get perplexity
        if (nodeData.model_metrics && nodeData.model_metrics.perplexity !== undefined) {
            perplexityValue = nodeData.model_metrics.perplexity;
        }

        // Get coherence
        if (nodeData.mallet_diagnostics && nodeData.mallet_diagnostics.coherence !== undefined) {
            coherenceValue = nodeData.mallet_diagnostics.coherence;
        }

        // If missing either metric, return gray
        if (perplexityValue === null || coherenceValue === null) {
            return "#999";
        }

        // Calculate normalized scores (0-1)
        const redIntensity = metricScales.perplexity(perplexityValue); // Low perplexity = high red
        const blueIntensity = metricScales.coherence(coherenceValue); // High coherence = high blue

        // Debug logging
        console.log(`${nodeId}: perp=${perplexityValue.toFixed(3)} (red=${redIntensity.toFixed(3)}), coh=${coherenceValue.toFixed(3)} (blue=${blueIntensity.toFixed(3)})`);

        // Ensure minimum brightness to avoid too dark colors
        const minBrightness = 0.2; // Minimum 20% brightness

        // Calculate color components with minimum brightness
        let red = 0, green = 0, blue = 0;
        if (colorMode === 'metric') {
            red = Math.round(255 * Math.max(minBrightness, redIntensity * metricConfig.red_weight));
            blue = Math.round(255 * Math.max(minBrightness, blueIntensity * metricConfig.blue_weight));
        } else if (colorMode === 'perplexity') {
            red = Math.round(255 * Math.max(minBrightness, redIntensity));
        } else if (colorMode === 'coherence') {
            blue = Math.round(255 * Math.max(minBrightness, blueIntensity));
        } else if (colorMode === 'metrics') {
            // 'metrics' mode: use coherence for node coloring (blue intensity)
            blue = Math.round(255 * Math.max(minBrightness, blueIntensity));
        }

        // Ensure values are in valid range
        const clampedRed = Math.max(0, Math.min(255, red));
        const clampedBlue = Math.max(0, Math.min(255, blue));
        const clampedGreen = 0;

        const finalColor = `rgb(${clampedRed}, ${clampedGreen}, ${clampedBlue})`;
        console.log(`${nodeId}: Final color = ${finalColor}`);

        return finalColor;
    }

    function calculateKLevelPerplexity(processedData, rawData) {
        console.log("Calculating K-level perplexity values...");
        
        const kPerplexityValues = {};
        
        // Extract one perplexity value per K level
        processedData.nodes.forEach(node => {
            const nodeData = rawData.nodes[node.id];
            if (nodeData && nodeData.model_metrics && nodeData.model_metrics.perplexity !== undefined) {
                const k = node.k;
                // Since all MCs within a K level should have the same perplexity, 
                // we only need to store it once per K
                if (kPerplexityValues[k] === undefined) {
                    kPerplexityValues[k] = nodeData.model_metrics.perplexity;
                    console.log(`K=${k}: Perplexity = ${nodeData.model_metrics.perplexity.toFixed(3)}`);
                }
            }
        });
        
        return kPerplexityValues;
    }

    function drawPerplexityBars(g, kValues, kPerplexityValues, metricScales, kSpacing, width) {
        if (!metricScales || !kPerplexityValues) {
            console.log("No metric scales or perplexity values available for bars");
            return;
        }
        
        console.log("Drawing perplexity bars above K labels...");
        
        // Bar dimensions - ensure bars don't go above SVG boundary
        const maxBarHeight = 35; // Reduced to prevent overflow
        const barWidth = Math.min(60, kSpacing * 0.8); // Adaptive width based on spacing
        const barY = -70; // Position above K labels, adjusted to prevent overflow
        
        // Find min/max perplexity for bar height scaling
        const perplexityValues = Object.values(kPerplexityValues);
        const minPerplexity = Math.min(...perplexityValues);
        const maxPerplexity = Math.max(...perplexityValues);
        const perplexityRange = maxPerplexity - minPerplexity;
        
        console.log(`Perplexity range for bars: ${minPerplexity.toFixed(3)} - ${maxPerplexity.toFixed(3)}`);
        
        // Draw bars for each K value
        kValues.forEach((k, index) => {
            const perplexityValue = kPerplexityValues[k];
            if (perplexityValue !== undefined) {
                
                // Calculate bar height (higher perplexity = taller bar)
                const normalizedHeight = perplexityRange > 0 ? 
                    (perplexityValue - minPerplexity) / perplexityRange : 0.5;
                // Ensure bar doesn't extend above the available space
                const barHeight = Math.max(5, Math.min(maxBarHeight, normalizedHeight * maxBarHeight));
                
                // Calculate color using same logic as perplexity mode
                const redIntensity = metricScales.perplexity(perplexityValue);
                const minBrightness = 0.2;
                const red = Math.round(255 * Math.max(minBrightness, redIntensity));
                const barColor = `rgb(${red}, 0, 0)`;
                
                const barX = index * kSpacing - barWidth / 2;
                
                // Draw the bar (no rounded corners)
                g.append("rect")
                    .attr("x", barX)
                    .attr("y", barY - barHeight)
                    .attr("width", barWidth)
                    .attr("height", barHeight)
                    .attr("fill", barColor)
                    .attr("stroke", "#333")
                    .attr("stroke-width", 1)
                    .style("cursor", "pointer")
                    .on("mouseover", function(event) {
                        // Show tooltip with perplexity value
                        const tooltip = g.append("g").attr("class", "perplexity-bar-tooltip");
                        const tooltipText = `K=${k}\\nPerplexity: ${perplexityValue.toFixed(3)}`;
                        const lines = tooltipText.split('\\n');
                        
                        const tooltipWidth = 120;
                        const tooltipHeight = lines.length * 15 + 10;
                        const tooltipX = barX - tooltipWidth/2 + barWidth/2;
                        const tooltipY = barY + 20; // Show below the bar instead of above
                        
                        tooltip.append("rect")
                            .attr("x", tooltipX)
                            .attr("y", tooltipY)
                            .attr("width", tooltipWidth)
                            .attr("height", tooltipHeight)
                            .attr("fill", "white")
                            .attr("stroke", "black")
                            .attr("rx", 3)
                            .attr("opacity", 0.9);
                        
                        lines.forEach((line, i) => {
                            tooltip.append("text")
                                .attr("x", tooltipX + tooltipWidth/2)
                                .attr("y", tooltipY + 15 + i * 15)
                                .attr("text-anchor", "middle")
                                .style("font-size", "11px")
                                .style("font-weight", "bold")
                                .style("fill", "black")
                                .text(line);
                        });
                    })
                    .on("mouseout", function() {
                        g.selectAll(".perplexity-bar-tooltip").remove();
                    });
                
                // Add perplexity value text below the bar
                g.append("text")
                    .attr("x", index * kSpacing)
                    .attr("y", barY + 15)
                    .attr("text-anchor", "middle")
                    .style("font-size", "10px")
                    .style("font-weight", "bold")
                    .style("fill", "#666")
                    .text(perplexityValue.toFixed(2));
            }
        });
        
        // Add title for the perplexity bars - positioned with more space to prevent overflow
        g.append("text")
            .attr("x", width / 2)
            .attr("y", barY - maxBarHeight - 5) // Increased space from -10 to -25
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text("Perplexity by K-value");
    }

    function drawColorLegend(g, width, height, metricMode, metricScales, colorMode) {
        // Remove any existing legends
        g.selectAll(".color-legend").remove();
        
        const legendX = 10;
        // Adjust Y position based on mode: metric mode needs more space
        const legendY = colorMode === 'metric' ? height - 180 : height - 120;
        
        const legend = g.append("g")
            .attr("class", "color-legend")
            .attr("transform", `translate(${legendX}, ${legendY})`);
        
        if (!metricMode) {
            // Default mode: K-value colors
            legend.append("text")
                .attr("x", 0)
                .attr("y", -5)
                .style("font-size", "12px")
                .style("font-weight", "bold")
                .style("fill", "#333")
                .text("Topic Colors by K-value");
        } else if (metricScales) {
            // Metric modes with interactive legends
            if (colorMode === 'perplexity') {
                drawPerplexityLegend(legend, metricScales, g);
            } else if (colorMode === 'coherence') {
                drawCoherenceLegend(legend, metricScales, g);
            } else if (colorMode === 'metric') {
                drawMetricLegend(legend, metricScales, g);
            } else if (colorMode === 'metrics') {
                // 'metrics' mode: nodes colored by coherence, bars show perplexity
                drawCoherenceLegend(legend, metricScales, g);
                // Add perplexity legend below coherence legend
                drawPerplexityLegendForMetrics(legend, metricScales, g);
            }
        }
    }
    
    function drawPerplexityLegend(legend, metricScales, g) {
        // Title
        legend.append("text")
            .attr("x", 0)
            .attr("y", -5)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text("Perplexity (Red Intensity)");
        
        // Create linear gradient for perplexity - access parent SVG correctly
        let svg = g;
        while (svg.node().tagName !== 'svg') {
            svg = d3.select(svg.node().parentNode);
        }
        const defs = svg.selectAll("defs").empty() ? 
                     svg.append("defs") : svg.select("defs");
        
        const gradientId = "perplexity-gradient";
        const gradient = defs.select(`#${gradientId}`).empty() ?
                        defs.append("linearGradient").attr("id", gradientId) :
                        defs.select(`#${gradientId}`);
        
        gradient
            .attr("x1", "0%")
            .attr("x2", "100%")
            .attr("y1", "0%")
            .attr("y2", "0%");
        
        gradient.selectAll("stop").remove();
        gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", "rgb(51, 0, 0)"); // Dark red (high perplexity = poor)
        gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", "rgb(255, 0, 0)"); // Bright red (low perplexity = good)
        
        const barWidth = 200;
        const barHeight = 15;
        
        // Draw gradient bar
        legend.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", barWidth)
            .attr("height", barHeight)
            .attr("fill", `url(#${gradientId})`)
            .attr("stroke", "#333")
            .attr("stroke-width", 1);
        
        // Add labels
        const extent = metricScales.perplexityExtent;
        legend.append("text")
            .attr("x", 0)
            .attr("y", barHeight + 12)
            .style("font-size", "10px")
            .style("fill", "#333")
            .text(`${extent[1].toFixed(2)} (poor)`);
        
        legend.append("text")
            .attr("x", barWidth)
            .attr("y", barHeight + 12)
            .attr("text-anchor", "end")
            .style("font-size", "10px")
            .style("fill", "#333")
            .text(`${extent[0].toFixed(2)} (good)`);
        
        // Create hover pointer
        const pointer = legend.append("g")
            .attr("class", "legend-pointer")
            .style("display", "none");
        
        pointer.append("line")
            .attr("x1", 0)
            .attr("x2", 0)
            .attr("y1", 10)
            .attr("y2", 35)
            .attr("stroke", "#ff6b35")
            .attr("stroke-width", 2);
        
        pointer.append("circle")
            .attr("cx", 0)
            .attr("cy", 22.5)
            .attr("r", 3)
            .attr("fill", "#ff6b35")
            .attr("stroke", "white")
            .attr("stroke-width", 1);
    }
    
    function drawCoherenceLegend(legend, metricScales, g) {
        // Title
        legend.append("text")
            .attr("x", 0)
            .attr("y", -5)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text("Coherence (Blue Intensity)");
        
        // Create linear gradient for coherence - access parent SVG correctly
        let svg = g;
        while (svg.node().tagName !== 'svg') {
            svg = d3.select(svg.node().parentNode);
        }
        const defs = svg.selectAll("defs").empty() ? 
                     svg.append("defs") : svg.select("defs");
        
        const gradientId = "coherence-gradient";
        const gradient = defs.select(`#${gradientId}`).empty() ?
                        defs.append("linearGradient").attr("id", gradientId) :
                        defs.select(`#${gradientId}`);
        
        gradient
            .attr("x1", "0%")
            .attr("x2", "100%")
            .attr("y1", "0%")
            .attr("y2", "0%");
        
        gradient.selectAll("stop").remove();
        gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", "rgb(0, 0, 51)"); // Dark blue (low coherence = poor)
        gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", "rgb(0, 0, 255)"); // Bright blue (high coherence = good)
        
        const barWidth = 200;
        const barHeight = 15;
        
        // Draw gradient bar
        legend.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", barWidth)
            .attr("height", barHeight)
            .attr("fill", `url(#${gradientId})`)
            .attr("stroke", "#333")
            .attr("stroke-width", 1);

        // Add labels
        const extent = metricScales.coherenceExtent;
        legend.append("text")
            .attr("x", 0)
            .attr("y", barHeight + 12)
            .attr("text-anchor", "start")
            .style("font-size", "10px")
            .style("fill", "#333")
            .text(`${extent[0].toFixed(2)} (poor)`);

        legend.append("text")
            .attr("x", barWidth)
            .attr("y", barHeight + 12)
            .attr("text-anchor", "end")
            .style("font-size", "10px")
            .style("fill", "#333")
            .text(`${extent[1].toFixed(2)} (good)`);
        
        // Create hover pointer
        const pointer = legend.append("g")
            .attr("class", "legend-pointer")
            .style("display", "none");
        
        pointer.append("line")
            .attr("x1", 0)
            .attr("x2", 0)
            .attr("y1", 10)
            .attr("y2", 35)
            .attr("stroke", "#ff6b35")
            .attr("stroke-width", 2);
        
        pointer.append("circle")
            .attr("cx", 0)
            .attr("cy", 22.5)
            .attr("r", 3)
            .attr("fill", "#ff6b35")
            .attr("stroke", "white")
            .attr("stroke-width", 1);
    }
    
    function drawPerplexityLegendForMetrics(legend, metricScales, g) {
        // Perplexity legend positioned below coherence legend
        const yOffset = 60; // Position below coherence legend
        
        const perplexityLegend = legend.append("g")
            .attr("class", "perplexity-legend-metrics")
            .attr("transform", `translate(0, ${yOffset})`);
            
        // Title
        perplexityLegend.append("text")
            .attr("x", 0)
            .attr("y", -5)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text("Perplexity (Bar Height)");
        
        // Create linear gradient for perplexity - access parent SVG correctly
        let svg = g;
        while (svg.node().tagName !== 'svg') {
            svg = d3.select(svg.node().parentNode);
        }
        const defs = svg.selectAll("defs").empty() ? 
                     svg.append("defs") : svg.select("defs");
        
        const gradientId = "perplexity-metrics-gradient";
        const gradient = defs.select(`#${gradientId}`).empty() ?
                        defs.append("linearGradient").attr("id", gradientId) :
                        defs.select(`#${gradientId}`);
        
        gradient
            .attr("x1", "0%")
            .attr("x2", "100%")
            .attr("y1", "0%")
            .attr("y2", "0%");
        
        gradient.selectAll("stop").remove();
        gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", "rgb(51, 0, 0)"); // Dark red (high perplexity = poor)
        gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", "rgb(255, 0, 0)"); // Bright red (low perplexity = good)
        
        const barWidth = 200;
        const barHeight = 15;
        
        // Draw gradient bar
        perplexityLegend.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", barWidth)
            .attr("height", barHeight)
            .attr("fill", `url(#${gradientId})`)
            .attr("stroke", "#333")
            .attr("stroke-width", 1);
        
        // Add labels
        const extent = metricScales.perplexityExtent;
        perplexityLegend.append("text")
            .attr("x", 0)
            .attr("y", barHeight + 12)
            .style("font-size", "10px")
            .style("fill", "#333")
            .text(`${extent[1].toFixed(2)} (poor)`);
        
        perplexityLegend.append("text")
            .attr("x", barWidth)
            .attr("y", barHeight + 12)
            .attr("text-anchor", "end")
            .style("font-size", "10px")
            .style("fill", "#333")
            .text(`${extent[0].toFixed(2)} (good)`);
    }
    
    function drawMetricLegend(legend, metricScales, g) {
        // Title
        legend.append("text")
            .attr("x", 0)
            .attr("y", -5)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text("Quality: Perplexity × Coherence");
        
        const rectSize = 100;
        
        // Create 2D gradient for metric mode - access parent SVG correctly
        let svg = g;
        while (svg.node().tagName !== 'svg') {
            svg = d3.select(svg.node().parentNode);
        }
        const defs = svg.selectAll("defs").empty() ? 
                     svg.append("defs") : svg.select("defs");
        
        // Create a pattern with multiple gradients to simulate 2D color space
        const patternId = "metric-pattern";
        const pattern = defs.select(`#${patternId}`).empty() ?
                       defs.append("pattern").attr("id", patternId) :
                       defs.select(`#${patternId}`);
        
        pattern
            .attr("patternUnits", "userSpaceOnUse")
            .attr("width", rectSize)
            .attr("height", rectSize);
        
        pattern.selectAll("*").remove();
        
        // Create color matrix for 2D space - FIXED for negative coherence values
        const resolution = 20;
        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const xPercent = i / (resolution - 1);  // X = perplexity (red) - left to right
                
                // For coherence: since higher (less negative) is better
                // Top of matrix should represent better coherence
                const yPercent = j / (resolution - 1);
                const coherencePercent = yPercent;  // Remove the (1 - yPercent)
                
                const red = Math.round(255 * Math.max(0.2, xPercent * 0.8));
                const blue = Math.round(255 * Math.max(0.2, coherencePercent * 0.8));
                
                pattern.append("rect")
                    .attr("x", (i / resolution) * rectSize)
                    .attr("y", (j / resolution) * rectSize)
                    .attr("width", rectSize / resolution + 0.5)  // Remove grid gaps
                    .attr("height", rectSize / resolution + 0.5) // Remove grid gaps
                    .attr("fill", `rgb(${red}, 0, ${blue})`)
                    .attr("stroke", "none");  // Remove grid lines
            }
        }
        
        // Draw the 2D color space
        legend.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", rectSize)
            .attr("height", rectSize)
            .attr("fill", `url(#${patternId})`)
            .attr("stroke", "#333")
            .attr("stroke-width", 1);
        
        // Add axis labels
        legend.append("text")
            .attr("x", rectSize / 2)
            .attr("y", rectSize + 15)
            .attr("text-anchor", "middle")
            .style("font-size", "10px")
            .style("fill", "#333")
            .text("Perplexity (Red) →");
        
        legend.append("text")
            .attr("x", -10)
            .attr("y", rectSize / 2)
            .attr("text-anchor", "middle")
            .attr("transform", `rotate(-90, -10, ${rectSize / 2})`)
            .style("font-size", "10px")
            .style("fill", "#333")
            .text("Coherence (Blue) →");
        
        // Create hover pointer for 2D space
        const pointer = legend.append("g")
            .attr("class", "legend-pointer")
            .style("display", "none");
        
        pointer.append("circle")
            .attr("cx", 0)
            .attr("cy", 0)
            .attr("r", 4)
            .attr("fill", "none")
            .attr("stroke", "#ff6b35")
            .attr("stroke-width", 2);
        
        pointer.append("circle")
            .attr("cx", 0)
            .attr("cy", 0)
            .attr("r", 2)
            .attr("fill", "#ff6b35");
    }
    
    function updateLegendPointer(nodeId, rawData, metricScales, colorMode) {
        if (!metricScales || !rawData.nodes[nodeId]) return;
        
        const nodeData = rawData.nodes[nodeId];
        let perplexityValue = null;
        let coherenceValue = null;
        
        if (nodeData.model_metrics && nodeData.model_metrics.perplexity !== undefined) {
            perplexityValue = nodeData.model_metrics.perplexity;
        }
        if (nodeData.mallet_diagnostics && nodeData.mallet_diagnostics.coherence !== undefined) {
            coherenceValue = nodeData.mallet_diagnostics.coherence;
        }
        
        if (perplexityValue === null || coherenceValue === null) return;
        
        // Find the correct legend pointer based on the color mode
        const pointer = d3.select(".color-legend .legend-pointer");
        if (pointer.empty()) return;
        
        pointer.style("display", "block");
        
        if (colorMode === 'perplexity') {
            const redIntensity = metricScales.perplexity(perplexityValue);
            const x = redIntensity * 200;
            pointer.attr("transform", `translate(${x}, 0)`);
        } else if (colorMode === 'coherence') {
            const blueIntensity = metricScales.coherence(coherenceValue);
            const x = blueIntensity * 200;
            pointer.attr("transform", `translate(${x}, 0)`);
        } else if (colorMode === 'metric') {
            const redIntensity = metricScales.perplexity(perplexityValue);
            const blueIntensity = metricScales.coherence(coherenceValue);
            const x = redIntensity * 100;
            const y = (1 - blueIntensity) * 100; // No offset needed since legend now starts at y=0
            pointer.attr("transform", `translate(${x}, ${y})`);
        }
    }
    
    function hideLegendPointer() {
        const pointer = d3.select(".color-legend .legend-pointer");
        if (!pointer.empty()) {
            pointer.style("display", "none");
        }
    }

    function drawMetricLegendOld(svg, metricScales, metricConfig, width, height, margin) {
        const legend = svg.append("g")
            .attr("class", "metric-legend")
            .attr("transform", `translate(${margin.left}, ${height - margin.bottom + 10})`);

        // Title
        legend.append("text")
            .attr("x", 0)
            .attr("y", 0)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text("Metric Mode: Perplexity (Red) × Coherence (Blue) = Quality (Purple)");

        // Color gradient demonstration
        const gradientWidth = 200;
        const gradientHeight = 15;

        // Create gradient definition
        const defs = svg.append("defs");

        const gradient = defs.append("linearGradient")
            .attr("id", "metric-gradient")
            .attr("x1", "0%")
            .attr("x2", "100%")
            .attr("y1", "0%")
            .attr("y2", "0%");

        // Add gradient stops to show the correct color mapping
        const stops = [
            { offset: "0%", color: "rgb(255, 0, 0)" },     // Pure red (high perplexity, low coherence)
            { offset: "25%", color: "rgb(200, 0, 55)" },   // Red-purple (high perplexity, medium coherence)  
            { offset: "50%", color: "rgb(128, 0, 128)" },  // Pure purple (medium perplexity, medium coherence)
            { offset: "75%", color: "rgb(55, 0, 200)" },   // Blue-purple (low perplexity, high coherence)
            { offset: "100%", color: "rgb(0, 0, 255)" }    // Pure blue (low perplexity, high coherence)
        ];

        stops.forEach(stop => {
            gradient.append("stop")
                .attr("offset", stop.offset)
                .attr("stop-color", stop.color);
        });

        // Draw gradient bar
        legend.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", gradientWidth)
            .attr("height", gradientHeight)
            .attr("fill", "url(#metric-gradient)")
            .attr("stroke", "#333")
            .attr("stroke-width", 1);

        // Add labels with correct interpretation
        legend.append("text")
            .attr("x", 0)
            .attr("y", 45)
            .style("font-size", "10px")
            .style("fill", "#d62728")
            .text("Poor Quality");

        legend.append("text")
            .attr("x", gradientWidth/2)
            .attr("y", 45)
            .attr("text-anchor", "middle")
            .style("font-size", "10px")
            .style("fill", "#7f4f7f")
            .text("Good Quality");

        legend.append("text")
            .attr("x", gradientWidth)
            .attr("y", 45)
            .attr("text-anchor", "end")
            .style("font-size", "10px")
            .style("fill", "#2f2fdf")
            .text("Excellent Quality");

        // Show current ranges with better formatting
        legend.append("text")
            .attr("x", gradientWidth + 20)
            .attr("y", 20)
            .style("font-size", "9px")
            .style("fill", "#666")
            .text(`Perplexity: ${metricScales.perplexityExtent[1].toFixed(2)} (poor) - ${metricScales.perplexityExtent[0].toFixed(2)} (good)`);

        legend.append("text")
            .attr("x", gradientWidth + 20)
            .attr("y", 35)
            .style("font-size", "9px")
            .style("fill", "#666")
            .text(`Coherence: ${metricScales.coherenceExtent[0].toFixed(2)} (poor) - ${metricScales.coherenceExtent[1].toFixed(2)} (good)`);
    }

    function processDataForVisualization(data) {
        const nodes = [];
        const flows = [];
        const kValues = data.k_range || [];

        // Process nodes - convert from dictionary to array
        Object.entries(data.nodes || {}).forEach(([nodeName, nodeData]) => {
            const match = nodeName.match(/K(\\d+)_MC(\\d+)/);
            if (match) {
                const k = parseInt(match[1]);
                const mc = parseInt(match[2]);

                nodes.push({
                    id: nodeName,
                    k: k,
                    mc: mc,
                    highCount: nodeData.high_count || 0,
                    mediumCount: nodeData.medium_count || 0,
                    totalProbability: nodeData.total_probability || 0,
                    highSamples: nodeData.high_samples || [],
                    mediumSamples: nodeData.medium_samples || []
                });
            }
        });

        // Process flows
        (data.flows || []).forEach(flow => {
            flows.push({
                source: flow.source_segment,
                target: flow.target_segment,
                sourceK: flow.source_k,
                targetK: flow.target_k,
                sampleCount: flow.sample_count || 0,
                averageProbability: flow.average_probability || 0,
                samples: flow.samples || []
            });
        });

        console.log(`Processed ${nodes.length} nodes and ${flows.length} flows`);
        return { nodes, flows, kValues };
    }

    function drawSankeyDiagram(g, data, width, height, colorSchemes, selectedFlow, model, metricMode, metricScales, metricConfig) {
        const { nodes, flows, kValues } = data;
        const rawData = model.get("sankey_data");

        if (nodes.length === 0) {
            g.append("text")
                .attr("x", width / 2)
                .attr("y", height / 2)
                .attr("text-anchor", "middle")
                .style("font-size", "16px")
                .style("fill", "#666")
                .text("No nodes to display");
            return;
        }

        // Filter flows - use configurable minimum flow threshold
        const minFlowSamples = model.get("min_flow_samples") || 10;
        const significantFlows = flows.filter(flow => flow.sampleCount >= minFlowSamples);
        console.log(`Showing ${significantFlows.length} flows out of ${flows.length} (filtered flows < ${minFlowSamples} samples)`);

        // Calculate positions with barycenter optimization
        const kSpacing = width / Math.max(1, kValues.length - 1);
        const nodesByK = d3.group(nodes, d => d.k);

        // Find max total count for scaling node heights
        const maxTotalCount = d3.max(nodes, d => d.highCount + d.mediumCount) || 1;
        const minNodeHeight = 20;
        const maxNodeHeight = 120;
        const minGap = 5; // Minimum gap between nodes
        
        // ADAPTIVE SCALING: Calculate required space for densest column
        console.log("=== Adaptive Node Scaling ===");
        
        // Calculate original node heights for all nodes
        nodes.forEach(node => {
            const totalSamples = node.highCount + node.mediumCount;
            node.originalHeight = minNodeHeight + (totalSamples / maxTotalCount) * (maxNodeHeight - minNodeHeight);
        });
        
        // Calculate space requirements for each K-level column
        const columnRequirements = {};
        kValues.forEach(k => {
            const kNodes = nodesByK.get(k) || [];
            let totalHeight = 0;
            
            // Sum up all node heights in this column
            kNodes.forEach(node => {
                totalHeight += node.originalHeight;
            });
            
            // Add gaps between nodes (n-1 gaps for n nodes)
            if (kNodes.length > 1) {
                totalHeight += (kNodes.length - 1) * minGap;
            }
            
            columnRequirements[k] = totalHeight;
            console.log(`K=${k}: ${kNodes.length} nodes, required height = ${totalHeight.toFixed(1)}px`);
        });
        
        // Find the densest column (requiring most space)
        const maxRequiredHeight = Math.max(...Object.values(columnRequirements));
        const availableHeight = height - 40; // Leave margin for labels
        
        console.log(`Densest column requires: ${maxRequiredHeight.toFixed(1)}px`);
        console.log(`Available height: ${availableHeight}px`);
        
        // Calculate scaling factor
        let scalingFactor = 1.0;
        let scaledMinHeight = minNodeHeight;
        let scaledMaxHeight = maxNodeHeight;
        let scaledGap = minGap;
        
        if (maxRequiredHeight > availableHeight) {
            scalingFactor = availableHeight / maxRequiredHeight;
            scaledMinHeight = minNodeHeight * scalingFactor;
            scaledMaxHeight = maxNodeHeight * scalingFactor;
            scaledGap = minGap * scalingFactor;
            
            console.log(`Scaling factor applied: ${scalingFactor.toFixed(3)}`);
            console.log(`Scaled min height: ${scaledMinHeight.toFixed(1)}px`);
            console.log(`Scaled max height: ${scaledMaxHeight.toFixed(1)}px`);
            console.log(`Scaled gap: ${scaledGap.toFixed(1)}px`);
        } else {
            console.log("No scaling needed - nodes fit perfectly!");
        }
        
        // Apply scaled heights to all nodes
        nodes.forEach(node => {
            const totalSamples = node.highCount + node.mediumCount;
            node.height = scaledMinHeight + (totalSamples / maxTotalCount) * (scaledMaxHeight - scaledMinHeight);
        });
        
        console.log("=== Adaptive Scaling Complete ===");

        // Apply barycenter method for node ordering (with updated scaled gap)
        const optimizedNodePositions = optimizeNodeOrderWithGap(nodes, significantFlows, kValues, nodesByK, height, scaledGap);

        // Position nodes using optimized order
        nodes.forEach(node => {
            const kIndex = kValues.indexOf(node.k);
            node.x = kIndex * kSpacing;
            node.y = optimizedNodePositions[node.id];
        });

        // Calculate flow width scaling
        const maxFlowCount = d3.max(significantFlows, d => d.sampleCount) || 1;
        const minFlowWidth = 2;
        const maxFlowWidth = 25;

        // Draw flows first (behind nodes)
        const flowGroup = g.append("g").attr("class", "flows");

        significantFlows.forEach((flow, flowIndex) => {
            // Parse source and target segment names
            const sourceTopicId = flow.source.replace(/_high$|_medium$/, '');
            const targetTopicId = flow.target.replace(/_high$|_medium$/, '');
            const sourceLevel = flow.source.includes('_high') ? 'high' : 'medium';
            const targetLevel = flow.target.includes('_high') ? 'high' : 'medium';

            const sourceNode = nodes.find(n => n.id === sourceTopicId);
            const targetNode = nodes.find(n => n.id === targetTopicId);

            if (sourceNode && targetNode && flow.sampleCount > 0) {
                // Proportional flow width scaling
                const flowWidth = minFlowWidth + (flow.sampleCount / maxFlowCount) * (maxFlowWidth - minFlowWidth);

                // Calculate connection points on the stacked bars
                const sourceY = calculateSegmentY(sourceNode, sourceLevel);
                const targetY = calculateSegmentY(targetNode, targetLevel);

                // Create curved path
                const curvePath = createCurvePath(
                    sourceNode.x + 15, sourceY,
                    targetNode.x - 15, targetY
                );

                // Check if this flow is selected
                const isSelected = selectedFlow && 
                    selectedFlow.source === flow.source && 
                    selectedFlow.target === flow.target &&
                    selectedFlow.sourceK === flow.sourceK &&
                    selectedFlow.targetK === flow.targetK;

                flowGroup.append("path")
                    .attr("d", curvePath)
                    .attr("stroke", isSelected ? "#ff6b35" : "#888")
                    .attr("stroke-width", isSelected ? flowWidth + 3 : flowWidth)
                    .attr("fill", "none")
                    .attr("opacity", isSelected ? 1.0 : 0.6)
                    .attr("class", `flow-${flowIndex}`)
                    .style("cursor", "pointer")
                    .on("mouseover", function(event) {
                        if (!isSelected) {
                            d3.select(this).attr("opacity", 0.8);
                        }
                        showTooltip(g, event, flow);
                    })
                    .on("mouseout", function() {
                        if (!isSelected) {
                            d3.select(this).attr("opacity", 0.6);
                        }
                        g.selectAll(".tooltip").remove();
                    })
                    .on("click", function(event) {
                        event.stopPropagation();
                        console.log("Flow clicked:", flow);

                        // Clear previous selection or select new flow
                        if (isSelected) {
                            model.set("selected_flow", {});
                        } else {
                            model.set("selected_flow", {
                                source: flow.source,
                                target: flow.target,
                                sourceK: flow.sourceK,
                                targetK: flow.targetK,
                                samples: flow.samples,
                                sampleCount: flow.sampleCount
                            });
                        }
                        model.save_changes();
                    });
            }
        });

        // Create sample tracing layer (initially empty)
        const tracingGroup = g.append("g").attr("class", "sample-tracing");

        // Draw nodes as stacked bars
        const nodeGroup = g.append("g").attr("class", "nodes");

        nodes.forEach(node => {
            const nodeG = nodeGroup.append("g")
                .attr("class", "node")
                .attr("transform", `translate(${node.x}, ${node.y - node.height/2})`);

            // Determine base color based on mode
            let baseColor;
            if (metricMode && metricScales) {
                baseColor = getMetricColor(node.id, rawData, metricScales, metricConfig, model.get("color_mode"));
            } else {
                baseColor = colorSchemes[node.k] || "#666";
            }

            // Calculate segment heights proportionally
            const totalCount = node.highCount + node.mediumCount;
            let highHeight = 0;
            let mediumHeight = 0;

            if (totalCount > 0) {
                highHeight = (node.highCount / totalCount) * node.height;
                mediumHeight = (node.mediumCount / totalCount) * node.height;
            }

            // In metric mode, use uniform colors; in default mode, use darker/lighter
            if (highHeight > 0) {
                const highColor = metricMode ? baseColor : d3.color(baseColor).darker(0.8);

                nodeG.append("rect")
                    .attr("x", -10)
                    .attr("y", 0)
                    .attr("width", 20)
                    .attr("height", highHeight)
                    .attr("fill", highColor)
                    .attr("stroke", "white")
                    .attr("stroke-width", 1)
                    .attr("class", `segment-${node.id}-high`)
                    .style("cursor", "pointer")
                    .on("mouseover", function(event) {
                        d3.select(this).attr("opacity", 0.8);
                        showSegmentTooltip(g, event, node, 'high', node.highCount, rawData, metricMode, model);
                        // Update legend pointer if in metric mode
                        if (metricMode && metricScales) {
                            updateLegendPointer(node.id, rawData, metricScales, model.get("color_mode"));
                        }
                    })
                    .on("mouseout", function() {
                        d3.select(this).attr("opacity", 1);
                        g.selectAll(".tooltip").remove();
                        // Hide legend pointer
                        if (metricMode) {
                            hideLegendPointer();
                        }
                    });
            }

            // Draw medium representation segment with hover
            if (mediumHeight > 0) {
                const mediumColor = metricMode ? baseColor : baseColor;

                nodeG.append("rect")
                    .attr("x", -10)
                    .attr("y", highHeight)
                    .attr("width", 20)
                    .attr("height", mediumHeight)
                    .attr("fill", mediumColor)
                    .attr("stroke", "white")
                    .attr("stroke-width", 1)
                    .attr("class", `segment-${node.id}-medium`)
                    .style("cursor", "pointer")
                    .on("mouseover", function(event) {
                        d3.select(this).attr("opacity", 0.8);
                        showSegmentTooltip(g, event, node, 'medium', node.mediumCount, rawData, metricMode, model);
                        // Update legend pointer if in metric mode
                        if (metricMode && metricScales) {
                            updateLegendPointer(node.id, rawData, metricScales, model.get("color_mode"));
                        }
                    })
                    .on("mouseout", function() {
                        d3.select(this).attr("opacity", 1);
                        g.selectAll(".tooltip").remove();
                        // Hide legend pointer
                        if (metricMode) {
                            hideLegendPointer();
                        }
                    });
            }

            // Add node label (only MC number, no sample count)
            nodeG.append("text")
                .attr("x", 25)
                .attr("y", node.height / 2)
                .attr("dy", "0.35em")
                .style("font-size", "11px")
                .style("font-weight", "bold")
                .style("fill", "#333")
                .text(`MC${node.mc}`)
                .style("cursor", "pointer")
                .on("click", function() {
                    console.log("Node clicked:", node);
                });
        });

        // Add click handler to clear selection when clicking on background
        g.on("click", function() {
            model.set("selected_flow", {});
            model.save_changes();
        });

        // Add K value labels at the top
        kValues.forEach((k, index) => {
            const labelColor = metricMode ? "#333" : (colorSchemes[k] || "#333");
            g.append("text")
                .attr("x", index * kSpacing)
                .attr("y", -30)
                .attr("text-anchor", "middle")
                .style("font-size", "16px")
                .style("font-weight", "bold")
                .style("fill", labelColor)
                .text(`K=${k}`);
        });

        // Draw perplexity bars for 'metrics' mode
        if (model.get("color_mode") === 'metrics' && metricMode && metricScales) {
            const kPerplexityValues = calculateKLevelPerplexity({nodes, flows, kValues}, rawData);
            drawPerplexityBars(g, kValues, kPerplexityValues, metricScales, kSpacing, width);
        }

        // Add color legend with hover functionality
        drawColorLegend(g, width, height, metricMode, metricScales, model.get("color_mode"));

        // Initial sample tracing if there's already a selected flow
        if (selectedFlow && Object.keys(selectedFlow).length > 0) {
            updateSampleTracing(g, data, selectedFlow, nodes, significantFlows, kValues, model);
        }

        // Draw all cluster trajectories if enabled
        // IMPORTANT: Use rawData (sankey_data) not processedData for cluster_trajectories
        if (model.get("show_all_clusters") && rawData.cluster_trajectories) {
            console.log("Initial cluster rendering - show_all_clusters is true");
            setGreyMode(g, true);  // Grey out nodes and flows
            drawAllClusterTrajectories(g, rawData, nodes, significantFlows, kValues, model);
        }

        // Listen for show_all_clusters changes
        model.on("change:show_all_clusters", () => {
            const showClusters = model.get("show_all_clusters");
            g.selectAll(".cluster-tracing").remove();
            g.selectAll(".cluster-legend").remove();
            
            if (showClusters && rawData.cluster_trajectories) {
                console.log("Cluster toggle - drawing clusters");
                setGreyMode(g, true);  // Grey out nodes and flows
                drawAllClusterTrajectories(g, rawData, nodes, significantFlows, kValues, model);
            } else {
                setGreyMode(g, false);  // Restore original colors
            }
        });
        
        // Listen for cluster_alpha changes to redraw with new transparency
        model.on("change:cluster_alpha", () => {
            if (model.get("show_all_clusters") && rawData.cluster_trajectories) {
                g.selectAll(".cluster-tracing").remove();
                g.selectAll(".cluster-legend").remove();
                console.log("Cluster alpha changed - redrawing clusters");
                drawAllClusterTrajectories(g, rawData, nodes, significantFlows, kValues, model);
            }
        });
    }
    
    // Function to toggle grey mode for nodes and flows
    function setGreyMode(g, enabled) {
        if (enabled) {
            // Grey out all node segments (two shades of grey)
            g.selectAll(".nodes rect").each(function() {
                const rect = d3.select(this);
                // Store original color if not already stored
                if (!rect.attr("data-original-fill")) {
                    rect.attr("data-original-fill", rect.attr("fill"));
                }
                // Use two shades of grey based on segment type (high vs medium)
                const className = rect.attr("class") || "";
                const isHigh = className.includes("-high");
                rect.attr("fill", isHigh ? "#999" : "#bbb");  // Darker grey for high, lighter for medium
            });
            
            // Grey out all flows
            g.selectAll(".flows path").each(function() {
                const path = d3.select(this);
                if (!path.attr("data-original-stroke")) {
                    path.attr("data-original-stroke", path.attr("stroke"));
                }
                path.attr("stroke", "#ddd");  // Light grey for flows
                path.attr("opacity", 0.3);     // More transparent
            });
            
            console.log("Grey mode enabled - nodes and flows greyed out");
        } else {
            // Restore original node colors
            g.selectAll(".nodes rect").each(function() {
                const rect = d3.select(this);
                const originalFill = rect.attr("data-original-fill");
                if (originalFill) {
                    rect.attr("fill", originalFill);
                }
            });
            
            // Restore original flow colors
            g.selectAll(".flows path").each(function() {
                const path = d3.select(this);
                const originalStroke = path.attr("data-original-stroke");
                if (originalStroke) {
                    path.attr("stroke", originalStroke);
                }
                path.attr("opacity", 0.6);  // Restore original opacity
            });
            
            console.log("Grey mode disabled - colors restored");
        }
    }

    function updateSampleTracing(g, data, selectedFlow, nodes, flows, kValues, model) {
        // Clear previous tracing
        g.selectAll(".sample-tracing").selectAll("*").remove();
        g.selectAll(".sample-count-badge").remove();
        g.selectAll(".sample-info-panel").remove();

        // Reset segment highlighting - set all segments back to white borders
        g.selectAll(".nodes rect").attr("stroke", "white").attr("stroke-width", 1);

        if (!selectedFlow || Object.keys(selectedFlow).length === 0) {
            return;
        }

        console.log("Tracing samples for selected flow:", selectedFlow);

        const tracingGroup = g.select(".sample-tracing");
        const samples = selectedFlow.samples || [];
        const sampleIds = samples.map(s => s.sample);

        console.log(`Tracing ${sampleIds.length} samples:`, sampleIds.slice(0, 3));

        if (sampleIds.length === 0) {
            showSampleInfo(g, selectedFlow, 0);
            return;
        }

        // Find where these samples are assigned across all K values
        const sampleAssignments = traceSampleAssignments(sampleIds, data, flows, kValues);

        // Draw sample trajectory paths with count-based line weights
        const minFlowSamples = model.get("min_flow_samples") || 10;
        drawSampleTrajectories(tracingGroup, sampleAssignments, nodes, selectedFlow, data, minFlowSamples);

        // Highlight segments containing these samples
        highlightSampleSegments(g, sampleAssignments, nodes);

        // Show detailed sample info panel
        showSampleInfo(g, selectedFlow, sampleIds.length);
    }

    function drawAllClusterTrajectories(g, rawData, nodes, flows, kValues, model) {
        console.log("Drawing all cluster trajectories...");
        console.log("rawData keys:", Object.keys(rawData));
        
        const clusterTrajectories = rawData.cluster_trajectories;
        if (!clusterTrajectories || clusterTrajectories.length === 0) {
            console.log("No cluster trajectories found in rawData");
            return;
        }
        
        console.log(`Found ${clusterTrajectories.length} cluster trajectories`);

        // Create a group for cluster trajectory visualization
        let clusterGroup = g.select(".cluster-tracing");
        if (clusterGroup.empty()) {
            clusterGroup = g.append("g").attr("class", "cluster-tracing");
        }
        clusterGroup.selectAll("*").remove();

        const minFlowSamples = model.get("min_flow_samples") || 10;

        // Calculate global max flow count for consistent scaling across all clusters
        // Use rawData.flows (raw format) not processedData flows
        const rawFlows = rawData.flows || [];
        const allFlows = rawFlows.filter(flow => (flow.sample_count || 0) >= minFlowSamples);
        const maxFlowCount = d3.max(allFlows, d => d.sample_count) || 1;
        const minFlowWidth = 1;
        const maxFlowWidth = 15;

        // Function to get line weight using same formula as sankey flows
        const getSankeyLineWeight = (count, clusterSize) => {
            // Scale by cluster size relative to max flow count
            const normalizedCount = count / clusterSize * maxFlowCount;
            return minFlowWidth + (normalizedCount / maxFlowCount) * (maxFlowWidth - minFlowWidth);
        };

        console.log(`Drawing ${clusterTrajectories.length} clusters`);

        // Draw each cluster's trajectories
        clusterTrajectories.forEach((cluster, clusterIndex) => {
            const clusterId = cluster.cluster_id;
            const baseColor = cluster.color; // rgba color string from data
            const sampleList = cluster.samples;
            const nSamples = cluster.n_samples;
            
            // Override alpha with widget's cluster_alpha setting
            const clusterAlpha = model.get("cluster_alpha") || 0.6;
            // Extract RGB from the base color and apply new alpha
            const rgbaMatch = baseColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
            const color = rgbaMatch 
                ? `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${clusterAlpha})`
                : baseColor;

            console.log(`Cluster ${clusterId}: ${nSamples} samples, color: ${color}, alpha: ${clusterAlpha}`);

            // Find sample assignments across K values
            // rawData is already passed to this function (it's the sankey_data)
            const sampleAssignments = traceSampleAssignmentsForCluster(sampleList, rawData, nodes, kValues);

            // Aggregate trajectories: count samples per segment transition
            const transitionCounts = {};
            const segmentCounts = {};

            Object.values(sampleAssignments).forEach(assignments => {
                const sortedKs = Object.keys(assignments).map(k => parseInt(k)).sort((a, b) => a - b);
                
                sortedKs.forEach(k => {
                    const assignment = assignments[k];
                    const segmentKey = `${assignment.topicId}-${assignment.level}`;
                    segmentCounts[segmentKey] = (segmentCounts[segmentKey] || 0) + 1;
                });

                for (let i = 0; i < sortedKs.length - 1; i++) {
                    const k1 = sortedKs[i];
                    const k2 = sortedKs[i + 1];
                    
                    // Only count adjacent K transitions
                    if (k2 - k1 === 1) {
                        const a1 = assignments[k1];
                        const a2 = assignments[k2];
                        const transitionKey = `${a1.topicId}-${a1.level}|${a2.topicId}-${a2.level}`;
                        transitionCounts[transitionKey] = (transitionCounts[transitionKey] || 0) + 1;
                    }
                }
            });

            // Draw aggregated trajectory paths (only if count >= minFlowSamples)
            let drawnPaths = 0;
            Object.entries(transitionCounts).forEach(([transitionKey, count]) => {
                // Apply same min_flow_samples filter as main Sankey flows
                if (count < minFlowSamples) {
                    return; // Skip small flows
                }
                
                const [fromKey, toKey] = transitionKey.split('|');
                const [fromTopic, fromLevel] = fromKey.split('-');
                const [toTopic, toLevel] = toKey.split('-');

                const fromNode = nodes.find(n => n.id === fromTopic);
                const toNode = nodes.find(n => n.id === toTopic);

                if (fromNode && toNode) {
                    const fromY = calculateSegmentY(fromNode, fromLevel);
                    const toY = calculateSegmentY(toNode, toLevel);

                    // Calculate line weight proportional to the count
                    const lineWeight = Math.max(1, Math.min(15, 1 + (count / nSamples) * 14));

                    const curvePath = createCurvePath(
                        fromNode.x + 15, fromY,
                        toNode.x - 15, toY
                    );

                    clusterGroup.append("path")
                        .attr("d", curvePath)
                        .attr("stroke", color)
                        .attr("stroke-width", lineWeight)
                        .attr("fill", "none")
                        .attr("opacity", 0.7)
                        .attr("class", `cluster-trajectory cluster-${clusterId}`)
                        .style("pointer-events", "none");
                    
                    drawnPaths++;
                }
            });
            
            console.log(`Cluster ${clusterId}: drew ${drawnPaths} paths (minFlowSamples=${minFlowSamples})`);

            // Draw dots at segment endpoints (only if count >= minFlowSamples)
            const significantSegments = Object.entries(segmentCounts).filter(([k, c]) => c >= minFlowSamples);
            const maxSegmentCount = Math.max(...significantSegments.map(([k, c]) => c), 1);
            
            significantSegments.forEach(([segmentKey, count]) => {
                const [topicId, level] = segmentKey.split('-');
                const node = nodes.find(n => n.id === topicId);
                
                if (node) {
                    const y = calculateSegmentY(node, level);
                    const dotRadius = 2 + (count / maxSegmentCount) * 4;

                    clusterGroup.append("circle")
                        .attr("cx", node.x)
                        .attr("cy", y)
                        .attr("r", dotRadius)
                        .attr("fill", color)
                        .attr("stroke", "white")
                        .attr("stroke-width", 0.5)
                        .attr("opacity", 0.8)
                        .attr("class", `cluster-point cluster-${clusterId}`)
                        .style("pointer-events", "none");
                }
            });
        });

        // Add cluster legend
        drawClusterLegend(g, clusterTrajectories, rawData.cluster_metadata);
        
        console.log("All cluster trajectories drawn");
    }

    function traceSampleAssignmentsForCluster(sampleIds, rawData, nodes, kValues) {
        const assignments = {};
        const sampleSet = new Set(sampleIds); // Use Set for faster lookup

        // Initialize assignment tracking for each sample
        sampleIds.forEach(sampleId => {
            assignments[sampleId] = {};
        });

        // IMPORTANT: Use raw data flows (data.flows), not filtered significantFlows
        // This ensures we capture all sample assignments even in smaller flows
        const rawFlows = rawData.flows || [];
        
        console.log(`Tracing ${sampleIds.length} cluster samples through ${rawFlows.length} raw flows`);

        rawFlows.forEach(flow => {
            const flowSamples = flow.samples || [];
            if (flowSamples.length > 0) {
                flowSamples.forEach(sampleData => {
                    const sampleId = sampleData.sample;

                    if (sampleSet.has(sampleId)) {
                        // Parse source segment: "K2_MC0_high" -> {topic: "K2_MC0", level: "high"}
                        const sourceSegment = flow.source_segment;
                        const targetSegment = flow.target_segment;
                        
                        const sourceTopicId = sourceSegment.replace(/_high$|_medium$/, '');
                        const sourceLevel = sourceSegment.includes('_high') ? 'high' : 'medium';
                        const targetTopicId = targetSegment.replace(/_high$|_medium$/, '');
                        const targetLevel = targetSegment.includes('_high') ? 'high' : 'medium';

                        assignments[sampleId][flow.source_k] = {
                            topicId: sourceTopicId,
                            level: sourceLevel,
                            probability: sampleData.source_prob || 0
                        };

                        assignments[sampleId][flow.target_k] = {
                            topicId: targetTopicId,
                            level: targetLevel,
                            probability: sampleData.target_prob || 0
                        };
                    }
                });
            }
        });

        // Count how many samples found assignments
        let samplesWithAssignments = 0;
        Object.values(assignments).forEach(a => {
            if (Object.keys(a).length > 0) samplesWithAssignments++;
        });
        console.log(`Found assignments for ${samplesWithAssignments}/${sampleIds.length} samples`);

        return assignments;
    }

    function drawClusterLegend(g, clusterTrajectories, metadata) {
        // Remove existing cluster legend
        g.selectAll(".cluster-legend").remove();

        const chartBounds = g.node().getBBox();
        const legendX = chartBounds.width + 20;
        const legendY = 10;

        const legend = g.append("g")
            .attr("class", "cluster-legend")
            .attr("transform", `translate(${legendX}, ${legendY})`);

        // Title
        legend.append("text")
            .attr("x", 0)
            .attr("y", 0)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text(`Clusters (${metadata.n_clusters})`);

        // Cluster entries
        clusterTrajectories.forEach((cluster, i) => {
            const entryY = 20 + i * 18;

            // Color swatch
            legend.append("rect")
                .attr("x", 0)
                .attr("y", entryY - 8)
                .attr("width", 12)
                .attr("height", 12)
                .attr("fill", cluster.color)
                .attr("stroke", "#333")
                .attr("stroke-width", 0.5);

            // Label
            legend.append("text")
                .attr("x", 18)
                .attr("y", entryY)
                .style("font-size", "10px")
                .style("fill", "#333")
                .text(`C${cluster.cluster_id} (${cluster.n_samples})`);
        });

        // Noise info if present
        if (metadata.n_noise > 0) {
            const noiseY = 20 + clusterTrajectories.length * 18;
            legend.append("text")
                .attr("x", 0)
                .attr("y", noiseY)
                .style("font-size", "9px")
                .style("fill", "#999")
                .text(`Noise: ${metadata.n_noise} samples`);
        }
    }

    function traceSampleAssignments(sampleIds, data, flows, kValues) {
        console.log("Tracing sample assignments across K values...");
        const assignments = {};

        // Initialize assignment tracking for each sample
        sampleIds.forEach(sampleId => {
            assignments[sampleId] = {};
        });

        // Go through all flows to find where samples appear
        flows.forEach(flow => {
            if (flow.samples && flow.samples.length > 0) {
                flow.samples.forEach(sampleData => {
                    const sampleId = sampleData.sample;

                    if (sampleIds.includes(sampleId)) {
                        // Extract topic and level from source segment
                        const sourceTopicId = flow.source.replace(/_high$|_medium$/, '');
                        const sourceLevel = flow.source.includes('_high') ? 'high' : 'medium';

                        // Extract topic and level from target segment  
                        const targetTopicId = flow.target.replace(/_high$|_medium$/, '');
                        const targetLevel = flow.target.includes('_high') ? 'high' : 'medium';

                        // Record source assignment
                        assignments[sampleId][flow.sourceK] = {
                            topicId: sourceTopicId,
                            level: sourceLevel,
                            probability: sampleData.source_prob || 0
                        };

                        // Record target assignment
                        assignments[sampleId][flow.targetK] = {
                            topicId: targetTopicId,
                            level: targetLevel,
                            probability: sampleData.target_prob || 0
                        };
                    }
                });
            }
        });

        console.log("Sample assignments traced:", Object.keys(assignments).length, "samples");
        return assignments;
    }

    function drawSampleTrajectories(tracingGroup, sampleAssignments, nodes, selectedFlow, data, minFlowSamples) {
        const trajectoryColor = "#ff6b35";
        const sampleIds = Object.keys(sampleAssignments);

        console.log(`Drawing trajectories for ${sampleIds.length} samples`);

        // First, calculate sample counts for each segment to determine line weights
        const segmentCounts = {};
        Object.values(sampleAssignments).forEach(assignments => {
            Object.values(assignments).forEach(assignment => {
                const segmentKey = `${assignment.topicId}-${assignment.level}`;
                segmentCounts[segmentKey] = (segmentCounts[segmentKey] || 0) + 1;
            });
        });

        // Use the SAME scaling as the main sankey diagram flows - using the passed configurable threshold
        const allFlows = data.flows.filter(flow => flow.sampleCount >= minFlowSamples);
        const maxFlowCount = d3.max(allFlows, d => d.sampleCount) || 1;
        const minFlowWidth = 2;
        const maxFlowWidth = 25;
    
        // Function to get line weight using same formula as sankey flows
        const getSankeyLineWeight = (count) => {
            return minFlowWidth + (count / maxFlowCount) * (maxFlowWidth - minFlowWidth);
        };

        sampleIds.forEach((sampleId, sampleIndex) => {
            const assignments = sampleAssignments[sampleId];
            const pathPoints = [];

            // Convert assignments to path points with coordinates
            Object.entries(assignments).forEach(([k, assignment]) => {
                const node = nodes.find(n => n.id === assignment.topicId);
                if (node) {
                    const segmentY = calculateSegmentY(node, assignment.level);
                    pathPoints.push({
                        k: parseInt(k),
                        x: node.x,
                        y: segmentY,
                        topicId: assignment.topicId,
                        level: assignment.level,
                        probability: assignment.probability,
                        sampleCount: segmentCounts[`${assignment.topicId}-${assignment.level}`] || 0
                    });
                }
            });

            // Sort path points by K value
            pathPoints.sort((a, b) => a.k - b.k);

            if (pathPoints.length >= 2) {
                // Draw trajectory ONLY between adjacent K values
                for (let i = 0; i < pathPoints.length - 1; i++) {
                    const start = pathPoints[i];
                    const end = pathPoints[i + 1];

                    // CRITICAL FIX: Only draw lines between adjacent K values
                    if (end.k - start.k === 1) {
                        // Check if this segment is the selected flow
                        const isSelectedSegment = 
                            start.k === selectedFlow.sourceK && 
                            end.k === selectedFlow.targetK;

                        // Calculate line weight using same scaling as sankey diagram
                        // Use the minimum sample count as the "flow capacity" between segments
                        const trajectoryFlowCount = Math.min(start.sampleCount, end.sampleCount);
                        const lineWeight = getSankeyLineWeight(trajectoryFlowCount);

                        const curvePath = createCurvePath(
                            start.x + 15, start.y,
                            end.x - 15, end.y
                        );

                        // ALL trajectory lines are now SOLID (no dashed lines)
                        tracingGroup.append("path")
                            .attr("d", curvePath)
                            .attr("stroke", trajectoryColor)
                            .attr("stroke-width", isSelectedSegment ? lineWeight + 2 : lineWeight)
                            .attr("stroke-dasharray", "none") // Always solid lines
                            .attr("fill", "none")
                            .attr("opacity", isSelectedSegment ? 0.9 : 0.7) // Slightly higher opacity for solid lines
                            .attr("class", `trajectory-${sampleIndex}-${i}`)
                            .style("pointer-events", "none");
                    }
                    // If end.k - start.k > 1, we skip drawing the line (gap in trajectory)
                }

                // Add dots at each assignment point (size proportional to sample count)
                const maxSampleCount = Math.max(...Object.values(segmentCounts));
                pathPoints.forEach((point, pointIndex) => {
                    // Scale dot size based on sample count using sankey proportions
                    const baseDotSize = 3;
                    const maxDotSize = 8; // Slightly larger to match sankey scale
                    const dotRadius = maxSampleCount > 0 ? 
                        baseDotSize + (point.sampleCount / maxSampleCount) * (maxDotSize - baseDotSize) : 
                        baseDotSize;

                    tracingGroup.append("circle")
                        .attr("cx", point.x)
                        .attr("cy", point.y)
                        .attr("r", dotRadius)
                        .attr("fill", trajectoryColor)
                        .attr("stroke", "white")
                        .attr("stroke-width", 1.5)
                        .attr("opacity", 0.8)
                        .attr("class", `trajectory-point-${sampleIndex}-${pointIndex}`)
                        .style("pointer-events", "none");
                });
            }
        });

        console.log("Sample trajectories drawn with sankey-matching line weights (all solid)");
    }

    function highlightSampleSegments(g, sampleAssignments, nodes) {
        const highlightColor = "#ff6b35";

        // Count how many samples are in each segment
        const segmentCounts = {};

        Object.values(sampleAssignments).forEach(assignments => {
            Object.values(assignments).forEach(assignment => {
                const segmentKey = `${assignment.topicId}-${assignment.level}`;
                segmentCounts[segmentKey] = (segmentCounts[segmentKey] || 0) + 1;
            });
        });

        console.log("Segment counts:", segmentCounts);

        // Highlight segments and add count badges
        Object.entries(segmentCounts).forEach(([segmentKey, count]) => {
            const [topicId, level] = segmentKey.split('-');

            // Highlight the segment with orange border
            g.selectAll(`.segment-${topicId}-${level}`)
                .attr("stroke", highlightColor)
                .attr("stroke-width", 3);

            // Find the node to position the count badge
            const node = nodes.find(n => n.id === topicId);
            if (node) {
                const badgeY = level === 'high' ? 
                    node.y - node.height/2 + 15 : 
                    node.y + node.height/2 - 15;

                // Add count badge
                g.append("circle")
                    .attr("cx", node.x + 35)
                    .attr("cy", badgeY)
                    .attr("r", 10)
                    .attr("fill", highlightColor)
                    .attr("stroke", "white")
                    .attr("stroke-width", 2)
                    .attr("class", "sample-count-badge");

                g.append("text")
                    .attr("x", node.x + 35)
                    .attr("y", badgeY)
                    .attr("text-anchor", "middle")
                    .attr("dy", "0.35em")
                    .style("font-size", "9px")
                    .style("font-weight", "bold")
                    .style("fill", "white")
                    .text(count)
                    .attr("class", "sample-count-badge");
            }
        });
    }

    function optimizeNodeOrder(nodes, flows, kValues, nodesByK, height) {
        console.log("Applying barycenter method for node ordering...");

        const nodePositions = {};

        // Step 1: Initialize first K level with even spacing
        const firstK = kValues[0];
        const firstKNodes = nodesByK.get(firstK) || [];
        const firstKSpacing = height / Math.max(1, firstKNodes.length + 1);

        firstKNodes.forEach((node, index) => {
            nodePositions[node.id] = (index + 1) * firstKSpacing;
        });

        console.log(`Initialized K=${firstK} with ${firstKNodes.length} nodes`);

        // Step 2: For each subsequent K level, calculate barycenter positions
        for (let kIndex = 1; kIndex < kValues.length; kIndex++) {
            const currentK = kValues[kIndex];
            const prevK = kValues[kIndex - 1];
            const currentKNodes = nodesByK.get(currentK) || [];

            console.log(`Optimizing K=${currentK} (${currentKNodes.length} nodes)`);

            // Calculate barycenter for each node in current K level
            const barycenterData = currentKNodes.map(node => {
                const nodeId = node.id;
                let weightedSum = 0;
                let totalWeight = 0;

                // Find all flows coming TO this node from previous K level
                flows.forEach(flow => {
                    const sourceTopicId = flow.source.replace(/_high$|_medium$/, '');
                    const targetTopicId = flow.target.replace(/_high$|_medium$/, '');

                    if (targetTopicId === nodeId && flow.sourceK === prevK) {
                        const sourcePosition = nodePositions[sourceTopicId];
                        if (sourcePosition !== undefined) {
                            const weight = flow.sampleCount;
                            weightedSum += sourcePosition * weight;
                            totalWeight += weight;
                        }
                    }
                });

                // Calculate barycenter (weighted average position)
                const barycenter = totalWeight > 0 ? weightedSum / totalWeight : height / 2;

                return {
                    node: node,
                    barycenter: barycenter,
                    totalWeight: totalWeight
                };
            });

            // Sort nodes by barycenter value
            barycenterData.sort((a, b) => a.barycenter - b.barycenter);

            // Assign positions based on sorted order with even spacing
            const kSpacing = height / Math.max(1, currentKNodes.length + 1);
            barycenterData.forEach((data, index) => {
                nodePositions[data.node.id] = (index + 1) * kSpacing;
            });
        }

        console.log("Barycenter optimization complete!");
        return nodePositions;
    }

    function optimizeNodeOrderWithGap(nodes, flows, kValues, nodesByK, height, scaledGap) {
        console.log("Applying gap-aware barycenter method for node ordering...");
        console.log(`Using scaled gap: ${scaledGap.toFixed(1)}px`);

        const nodePositions = {};
        const marginTop = 20; // Top margin

        // Step 1: Initialize first K level with proper gap-aware spacing
        const firstK = kValues[0];
        const firstKNodes = nodesByK.get(firstK) || [];
        
        // Calculate total height needed for first K level
        let totalFirstKHeight = 0;
        firstKNodes.forEach(node => {
            totalFirstKHeight += node.height;
        });
        totalFirstKHeight += (firstKNodes.length - 1) * scaledGap;
        
        // Position first K level nodes with proper gaps
        let currentY = marginTop;
        firstKNodes.forEach((node, index) => {
            nodePositions[node.id] = currentY + node.height / 2;
            currentY += node.height + scaledGap;
        });

        console.log(`Initialized K=${firstK} with ${firstKNodes.length} nodes using gap-aware spacing`);

        // Step 2: For each subsequent K level, calculate barycenter positions with gap awareness
        for (let kIndex = 1; kIndex < kValues.length; kIndex++) {
            const currentK = kValues[kIndex];
            const prevK = kValues[kIndex - 1];
            const currentKNodes = nodesByK.get(currentK) || [];

            console.log(`Optimizing K=${currentK} (${currentKNodes.length} nodes)`);

            // Calculate barycenter for each node in current K level
            const barycenterData = currentKNodes.map(node => {
                const nodeId = node.id;
                let weightedSum = 0;
                let totalWeight = 0;

                // Find all flows coming TO this node from previous K level
                flows.forEach(flow => {
                    const sourceTopicId = flow.source.replace(/_high$|_medium$/, '');
                    const targetTopicId = flow.target.replace(/_high$|_medium$/, '');

                    if (targetTopicId === nodeId && flow.sourceK === prevK) {
                        const sourcePosition = nodePositions[sourceTopicId];
                        if (sourcePosition !== undefined) {
                            const weight = flow.sampleCount;
                            weightedSum += sourcePosition * weight;
                            totalWeight += weight;
                        }
                    }
                });

                // Calculate barycenter (weighted average position)
                const barycenter = totalWeight > 0 ? weightedSum / totalWeight : height / 2;

                return {
                    node: node,
                    barycenter: barycenter,
                    totalWeight: totalWeight
                };
            });

            // Sort nodes by barycenter value
            barycenterData.sort((a, b) => a.barycenter - b.barycenter);

            // Position nodes with proper gap-aware spacing
            let currentY = marginTop;
            barycenterData.forEach((data, index) => {
                nodePositions[data.node.id] = currentY + data.node.height / 2;
                currentY += data.node.height + scaledGap;
            });
        }

        console.log("Gap-aware barycenter optimization complete!");
        return nodePositions;
    }

    function calculateSegmentY(node, level) {
        const totalCount = node.highCount + node.mediumCount;
        if (totalCount === 0) return node.y;

        const highHeight = (node.highCount / totalCount) * node.height;

        if (level === 'high') {
            return node.y - node.height/2 + highHeight/2;
        } else {
            return node.y - node.height/2 + highHeight + (node.height - highHeight)/2;
        }
    }

    function createCurvePath(x1, y1, x2, y2) {
        const midX = (x1 + x2) / 2;
        return `M ${x1} ${y1} C ${midX} ${y1} ${midX} ${y2} ${x2} ${y2}`;
    }

    function showTooltip(g, event, flow) {
        const tooltip = g.append("g").attr("class", "tooltip");

        const tooltipText = `${flow.sampleCount} samples\\n${flow.source} → ${flow.target}`;
        const lines = tooltipText.split('\\n');

        const tooltipWidth = 160;
        const tooltipHeight = 35;

        // Get the chart dimensions to ensure tooltip stays within bounds
        const chartWidth = g.node().getBBox().width || 1000;

        // Calculate tooltip position with bounds checking - show below by default
        let tooltipX = event.layerX || 0;
        let tooltipY = (event.layerY || 0) + 20; // Show below cursor by default

        // Adjust X position if tooltip would go off the right edge
        if (tooltipX + tooltipWidth > chartWidth) {
            tooltipX = chartWidth - tooltipWidth - 10;
        }

        // Get chart bounds to check bottom edge
        const chartBounds = g.node().getBBox();
        const chartHeight = chartBounds.height || 600;
        
        // Adjust Y position if tooltip would go off the bottom edge
        if (tooltipY + tooltipHeight > chartHeight) {
            tooltipY = (event.layerY || 0) - tooltipHeight - 10; // Show above cursor if no room below
        }

        const rect = tooltip.append("rect")
            .attr("x", tooltipX)
            .attr("y", tooltipY)
            .attr("width", tooltipWidth)
            .attr("height", tooltipHeight)
            .attr("fill", "white")
            .attr("stroke", "black")
            .attr("rx", 3)
            .attr("opacity", 0.9);

        lines.forEach((line, i) => {
            tooltip.append("text")
                .attr("x", tooltipX + 5)
                .attr("y", tooltipY + 15 + i * 12)
                .style("font-size", "10px")
                .style("fill", "black")
                .text(line);
        });
    }

    function showSampleInfo(g, selectedFlow, sampleCount) {
        // Position the info panel above the legend with a small gap
        const chartBounds = g.node().getBBox();
        const panelHeight = 60;
        
        // Calculate position based on legend position
        // Legends are positioned at different heights based on color mode
        const colorLegend = g.select(".color-legend");
        let legendY = chartBounds.height - 120; // Default legend position
        
        if (!colorLegend.empty()) {
            // Get the actual legend position from its transform
            const legendTransform = colorLegend.attr("transform");
            const matches = legendTransform.match(/translate\((\d+),\s*(\d+)\)/);
            if (matches) {
                legendY = parseInt(matches[2]);
            }
        }
        
        // Position panel below legend with 20px gap
        // Add space for legend content - estimate legend height based on color mode
        let legendHeight = 40; // Default legend height
        if (!colorLegend.empty()) {
            // For different legend types, estimate the height
            if (g.select(".perplexity-legend-metrics").empty() === false) {
                legendHeight = 120; // metrics mode has two legends stacked
            } else {
                legendHeight = 60; // single legend height
            }
        }
        const panelY = legendY + legendHeight -10;
        
        const infoPanel = g.append("g").attr("class", "sample-info-panel");

        // Background panel positioned at bottom-left
        infoPanel.append("rect")
            .attr("x", 10)
            .attr("y", panelY)
            .attr("width", 200)
            .attr("height", panelHeight)
            .attr("fill", "white")
            .attr("stroke", "#ff6b35")
            .attr("stroke-width", 2)
            .attr("rx", 5)
            .attr("opacity", 0.95);

        // Title
        infoPanel.append("text")
            .attr("x", 20)
            .attr("y", panelY + 20)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .style("fill", "#ff6b35")
            .text(`Selected: ${sampleCount} Samples`);

        // Flow info
        infoPanel.append("text")
            .attr("x", 20)
            .attr("y", panelY + 35)
            .style("font-size", "10px")
            .style("fill", "#333")
            .text(`${selectedFlow.source} → ${selectedFlow.target}`);

        // Instructions
        infoPanel.append("text")
            .attr("x", 20)
            .attr("y", panelY + 48)
            .style("font-size", "9px")
            .style("fill", "#666")
            .text("Click flow again or background to clear");
    }

    function showSegmentTooltip(g, event, node, level, count, rawData, metricMode, model) {
        const tooltip = g.append("g").attr("class", "tooltip");

        // Get configurable thresholds from the model
        const thresholds = model.get("probability_thresholds") || { high_threshold: 0.67, medium_threshold: 0.33 };
        const highThreshold = thresholds.high_threshold;
        const mediumThreshold = thresholds.medium_threshold;
        
        const levelText = level === 'high' ? 
            `High (≥${highThreshold})` : 
            `Medium (${mediumThreshold}-${highThreshold.toFixed(2)})`;
        let tooltipLines = [`${node.id}`, levelText, `${count} samples`];

        // Add metric information if in metric mode
        if (metricMode && rawData && rawData.nodes[node.id]) {
            const nodeData = rawData.nodes[node.id];

            if (nodeData.model_metrics && nodeData.model_metrics.perplexity !== undefined) {
                tooltipLines.push(`Perplexity: ${nodeData.model_metrics.perplexity.toFixed(3)}`);
            }

            if (nodeData.mallet_diagnostics && nodeData.mallet_diagnostics.coherence !== undefined) {
                tooltipLines.push(`Coherence: ${nodeData.mallet_diagnostics.coherence.toFixed(3)}`);
            }
        }

        const tooltipHeight = tooltipLines.length * 12 + 10;
        const tooltipWidth = Math.max(140, Math.max(...tooltipLines.map(line => line.length * 6 + 10)));

        // Get the chart dimensions to ensure tooltip stays within bounds
        const chartWidth = g.node().getBBox().width || 1000;

        // Calculate tooltip position relative to the node's actual position
        // Use node position instead of event coordinates for more accurate positioning
        let tooltipX = node.x + 25; // Position to the right of the node
        let tooltipY = node.y + 10; // Slightly below node center

        // Adjust X position if tooltip would go off the right edge
        if (tooltipX + tooltipWidth > chartWidth) {
            tooltipX = chartWidth - tooltipWidth - 10;
        }

        // Get chart bounds to check bottom edge
        const chartBounds = g.node().getBBox();
        const chartHeight = chartBounds.height || 600;
        
        // Adjust Y position if tooltip would go off the bottom edge
        if (tooltipY + tooltipHeight > chartHeight) {
            tooltipY = (event.layerY || 0) - tooltipHeight - 10; // Show above cursor if no room below
        }

        const rect = tooltip.append("rect")
            .attr("x", tooltipX)
            .attr("y", tooltipY)
            .attr("width", tooltipWidth)
            .attr("height", tooltipHeight)
            .attr("fill", "white")
            .attr("stroke", "black")
            .attr("rx", 3)
            .attr("opacity", 0.9);

        tooltipLines.forEach((line, i) => {
            tooltip.append("text")
                .attr("x", tooltipX + 5)
                .attr("y", tooltipY + 15 + i * 12)
                .style("font-size", "10px")
                .style("fill", "black")
                .text(line);
        });
    }

    export default { render };
    """

    _css = """
    .widget-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    }

    .sample-tracing {
        pointer-events: none;
    }

    .sample-info-panel {
        pointer-events: none;
    }

    .metric-legend {
        pointer-events: none;
    }
    """

    # Widget traits
    sankey_data = traitlets.Dict(default_value={}).tag(sync=True)
    width = traitlets.Int(default_value=1200).tag(sync=True)
    height = traitlets.Int(default_value=800).tag(sync=True)

    # Add trait for tracking selected flow
    selected_flow = traitlets.Dict(default_value={}).tag(sync=True)

    # Add traits for metric mode
    metric_mode = traitlets.Bool(default_value=False).tag(sync=True)
    color_mode = traitlets.Unicode(default_value="default").tag(sync=True)  # "default", "metric", "perplexity", "coherence", "metrics"
    metric_config = traitlets.Dict(default_value={
        'red_weight': 0.8,    # Weight for perplexity (red component)
        'blue_weight': 0.8,   # Weight for coherence (blue component)
        'min_saturation': 0.3  # Minimum color saturation to keep colors visible
    }).tag(sync=True)
    
    # Add traits for probability thresholds
    probability_thresholds = traitlets.Dict(default_value={
        'high_threshold': 0.67,   # Threshold for high representation (≥ this value)
        'medium_threshold': 0.33  # Threshold for medium representation (≥ this value)
    }).tag(sync=True)
    
    # Add trait for minimum flow threshold
    min_flow_samples = traitlets.Int(default_value=10).tag(sync=True)  # Minimum samples required to show a flow
    
    # Add trait for showing all cluster trajectories
    show_all_clusters = traitlets.Bool(default_value=False).tag(sync=True)
    
    # Add trait for cluster trajectory transparency (0.0 - 1.0)
    cluster_alpha = traitlets.Float(default_value=0.6).tag(sync=True)

    color_schemes = traitlets.Dict(default_value={
        2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c", 5: "#d62728", 6: "#9467bd",
        7: "#8c564b", 8: "#e377c2", 9: "#7f7f7f", 10: "#bcbd22"
    }).tag(sync=True)

    def __init__(self, sankey_data=None, mode="default", high_threshold=None, medium_threshold=None, min_flow_samples=None, show_clusters=False, cluster_alpha=0.6, **kwargs):
        super().__init__(**kwargs)
        if sankey_data:
            self.sankey_data = sankey_data
        # Set metric_mode based on the mode parameter
        self.metric_mode = (mode in ["metric", "perplexity", "coherence", "metrics"])
        # Store the specific mode type
        self.color_mode = mode
        
        # Set custom thresholds if provided
        if high_threshold is not None or medium_threshold is not None:
            thresholds = self.probability_thresholds.copy()
            if high_threshold is not None:
                thresholds['high_threshold'] = high_threshold
            if medium_threshold is not None:
                thresholds['medium_threshold'] = medium_threshold
            self.probability_thresholds = thresholds
        
        # Set custom minimum flow samples threshold if provided
        if min_flow_samples is not None:
            self.min_flow_samples = min_flow_samples
        
        # Set show_all_clusters if provided
        self.show_all_clusters = show_clusters
        
        # Set cluster transparency
        self.cluster_alpha = cluster_alpha

    def set_mode(self, mode):
        """Set visualization mode: 'default', 'metric', 'perplexity', 'coherence', or 'metrics'"""
        self.metric_mode = (mode in ["metric", "perplexity", "coherence", "metrics"])
        self.color_mode = mode
        return self  # Return self for chaining

    def update_metric_config(self, red_weight=None, blue_weight=None, min_saturation=None):
        """Update metric mode configuration"""
        config = self.metric_config.copy()
        if red_weight is not None:
            config['red_weight'] = red_weight
        if blue_weight is not None:
            config['blue_weight'] = blue_weight
        if min_saturation is not None:
            config['min_saturation'] = min_saturation
        self.metric_config = config
        return self  # Return self for chaining
    
    def update_probability_thresholds(self, high_threshold=None, medium_threshold=None):
        """Update probability thresholds for high and medium representation levels"""
        thresholds = self.probability_thresholds.copy()
        if high_threshold is not None:
            thresholds['high_threshold'] = high_threshold
        if medium_threshold is not None:
            thresholds['medium_threshold'] = medium_threshold
        self.probability_thresholds = thresholds
        return self  # Return self for chaining
    
    def toggle_clusters(self, show=None):
        """Toggle or set cluster trajectory visualization.
        
        Parameters
        ----------
        show : bool, optional
            If provided, set to this value. If None, toggle current state.
        
        Returns
        -------
        self for chaining
        """
        if show is None:
            self.show_all_clusters = not self.show_all_clusters
        else:
            self.show_all_clusters = show
        return self
