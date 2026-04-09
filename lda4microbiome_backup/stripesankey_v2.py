"""
stripesankey_v2.py - Improved StripeSankey with proper Sankey-style flow stacking.

Changes from v1 (StripeSankeyInline):
- Flows are drawn as filled ribbons that stack along segment edges
- Flow width is proportional to the segment height (true Sankey behaviour)
- Flows are sorted to minimise crossings (by target/source Y position)
- Ribbon paths taper between source and target (variable width)
- All original functions (metrics, legends, clusters, sample tracing) are preserved

Usage: drop-in replacement for StripeSankeyInline.

    from lda4microbiome_backup.stripesankey_v2 import StripeSankeyV2
    widget = StripeSankeyV2(sankey_data=sankey_data, width=1400, height=700)
"""

from .stripesankey import StripeSankeyInline

# ---------------------------------------------------------------------------
# New JavaScript helper functions (injected into the ESM)
# ---------------------------------------------------------------------------

_NEW_JS_FUNCTIONS = r"""
    // === V2 helpers: proper Sankey flow stacking ===

    /**
     * Return the absolute top / bottom Y coordinates of a segment
     * (high or medium) within a node.
     */
    function getSegmentBounds(node, level) {
        const totalCount = node.highCount + node.mediumCount;
        if (totalCount === 0) return { top: node.y, bottom: node.y };

        const nodeTop = node.y - node.height / 2;
        const highHeight = (node.highCount / totalCount) * node.height;

        if (level === 'high') {
            return { top: nodeTop, bottom: nodeTop + highHeight };
        } else {
            return { top: nodeTop + highHeight, bottom: nodeTop + node.height };
        }
    }

    /**
     * Pre-compute a stacked layout for every significant flow.
     *
     * Returns a Map keyed by "source|target|sourceK|targetK" with values:
     *   { sourceYTop, sourceYBot, targetYTop, targetYBot }
     *
     * Flows within each segment are sorted so that flows heading toward
     * higher-Y targets are placed at the bottom of the source segment
     * (and vice-versa), which reduces ribbon crossings.
     */
    function computeFlowLayout(nodes, significantFlows) {
        // Fast node lookup
        const nodeMap = new Map();
        nodes.forEach(n => nodeMap.set(n.id, n));

        // ---- 1. Group flows by segment ----
        const outFlows = {};   // source segment key -> [flow, ...]
        const inFlows  = {};   // target segment key -> [flow, ...]

        significantFlows.forEach(flow => {
            (outFlows[flow.source] || (outFlows[flow.source] = [])).push(flow);
            (inFlows[flow.target]  || (inFlows[flow.target]  = [])).push(flow);
        });

        // Helper: centre-Y of a segment (used for sorting)
        function segCentreY(segKey) {
            const topicId = segKey.replace(/_high$|_medium$/, '');
            const level   = segKey.includes('_high') ? 'high' : 'medium';
            const node    = nodeMap.get(topicId);
            if (!node) return 0;
            return calculateSegmentY(node, level);   // original v1 helper
        }

        // ---- 2. Sort to minimise crossings ----
        Object.values(outFlows).forEach(flows => {
            flows.sort((a, b) => segCentreY(a.target) - segCentreY(b.target));
        });
        Object.values(inFlows).forEach(flows => {
            flows.sort((a, b) => segCentreY(a.source) - segCentreY(b.source));
        });

        // ---- 3. Totals per segment ----
        const outTotals = {};
        const inTotals  = {};
        Object.entries(outFlows).forEach(([k, fs]) => {
            outTotals[k] = fs.reduce((s, f) => s + f.sampleCount, 0);
        });
        Object.entries(inFlows).forEach(([k, fs]) => {
            inTotals[k] = fs.reduce((s, f) => s + f.sampleCount, 0);
        });

        // ---- 4. Assign source-side Y offsets ----
        const srcLayout = {};
        Object.entries(outFlows).forEach(([segKey, flows]) => {
            const topicId = segKey.replace(/_high$|_medium$/, '');
            const level   = segKey.includes('_high') ? 'high' : 'medium';
            const node    = nodeMap.get(topicId);
            if (!node) return;
            const bounds = getSegmentBounds(node, level);
            const segH   = bounds.bottom - bounds.top;
            const total  = outTotals[segKey] || 1;
            let off = 0;
            flows.forEach(flow => {
                const h  = (flow.sampleCount / total) * segH;
                const fk = flow.source + '|' + flow.target + '|' + flow.sourceK + '|' + flow.targetK;
                srcLayout[fk] = { yTop: bounds.top + off, yBot: bounds.top + off + h };
                off += h;
            });
        });

        // ---- 5. Assign target-side Y offsets ----
        const tgtLayout = {};
        Object.entries(inFlows).forEach(([segKey, flows]) => {
            const topicId = segKey.replace(/_high$|_medium$/, '');
            const level   = segKey.includes('_high') ? 'high' : 'medium';
            const node    = nodeMap.get(topicId);
            if (!node) return;
            const bounds = getSegmentBounds(node, level);
            const segH   = bounds.bottom - bounds.top;
            const total  = inTotals[segKey] || 1;
            let off = 0;
            flows.forEach(flow => {
                const h  = (flow.sampleCount / total) * segH;
                const fk = flow.source + '|' + flow.target + '|' + flow.sourceK + '|' + flow.targetK;
                tgtLayout[fk] = { yTop: bounds.top + off, yBot: bounds.top + off + h };
                off += h;
            });
        });

        // ---- 6. Combine into final Map ----
        const result = new Map();
        significantFlows.forEach(flow => {
            const fk  = flow.source + '|' + flow.target + '|' + flow.sourceK + '|' + flow.targetK;
            const src = srcLayout[fk];
            const tgt = tgtLayout[fk];
            if (src && tgt) {
                result.set(fk, {
                    sourceYTop: src.yTop,
                    sourceYBot: src.yBot,
                    targetYTop: tgt.yTop,
                    targetYBot: tgt.yBot
                });
            }
        });

        console.log(`V2 flow layout computed: ${result.size} flows stacked`);
        return result;
    }

    /**
     * Create a filled ribbon path that can taper from source width
     * to a different target width.
     *
     *   (x1, y1Top)-----bezier----->(x2, y2Top)
     *        |                            |
     *   (x1, y1Bot)<----bezier------(x2, y2Bot)
     */
    function createRibbonPath(x1, y1Top, y1Bot, x2, y2Top, y2Bot) {
        const midX = (x1 + x2) / 2;
        return 'M ' + x1 + ' ' + y1Top
             + ' C ' + midX + ' ' + y1Top + ' ' + midX + ' ' + y2Top + ' ' + x2 + ' ' + y2Top
             + ' L ' + x2 + ' ' + y2Bot
             + ' C ' + midX + ' ' + y2Bot + ' ' + midX + ' ' + y1Bot + ' ' + x1 + ' ' + y1Bot
             + ' Z';
    }
"""

# ---------------------------------------------------------------------------
# Replacement flow-drawing code (replaces the stroke-width loop in v1)
# ---------------------------------------------------------------------------

_NEW_FLOW_CODE = r"""// === V2: Stacked flow layout with filled ribbons ===
        const flowLayout = computeFlowLayout(nodes, significantFlows);

        // Draw flows as filled ribbons (behind nodes)
        const flowGroup = g.append("g").attr("class", "flows");

        significantFlows.forEach((flow, flowIndex) => {
            const sourceTopicId = flow.source.replace(/_high$|_medium$/, '');
            const targetTopicId = flow.target.replace(/_high$|_medium$/, '');

            const sourceNode = nodes.find(n => n.id === sourceTopicId);
            const targetNode = nodes.find(n => n.id === targetTopicId);

            if (sourceNode && targetNode && flow.sampleCount > 0) {
                const fk = flow.source + '|' + flow.target + '|' + flow.sourceK + '|' + flow.targetK;
                const layout = flowLayout.get(fk);
                if (!layout) return;

                // Build filled ribbon path (variable width from source to target)
                const ribbonPath = createRibbonPath(
                    sourceNode.x + 15, layout.sourceYTop, layout.sourceYBot,
                    targetNode.x - 15, layout.targetYTop, layout.targetYBot
                );

                // Check if this flow is selected
                const isSelected = selectedFlow &&
                    selectedFlow.source === flow.source &&
                    selectedFlow.target === flow.target &&
                    selectedFlow.sourceK === flow.sourceK &&
                    selectedFlow.targetK === flow.targetK;

                const defaultFill    = "rgba(180, 180, 180, 0.40)";
                const defaultStroke  = "rgba(150, 150, 150, 0.55)";
                const selectedFill   = "rgba(255, 107, 53, 0.65)";
                const selectedStroke = "rgba(200, 80, 30, 0.80)";
                const hoverFill      = "rgba(160, 160, 160, 0.55)";

                flowGroup.append("path")
                    .attr("d", ribbonPath)
                    .attr("data-flow-key", fk)
                    .attr("fill",   isSelected ? selectedFill   : defaultFill)
                    .attr("stroke", isSelected ? selectedStroke : defaultStroke)
                    .attr("stroke-width", 0.5)
                    .attr("opacity", isSelected ? 0.9 : 1.0)
                    .attr("class", `flow-${flowIndex} flow-ribbon`)
                    .style("cursor", "pointer")
                    .on("mouseover", function(event) {
                        if (!isSelected) {
                            d3.select(this).attr("fill", hoverFill);
                        }
                        showTooltip(g, event, flow);
                    })
                    .on("mouseout", function() {
                        if (!isSelected) {
                            d3.select(this).attr("fill", defaultFill);
                        }
                        g.selectAll(".tooltip").remove();
                    })
                    .on("click", function(event) {
                        event.stopPropagation();
                        console.log("Flow clicked:", flow);

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

        """

# ---------------------------------------------------------------------------
# Replacement drawSampleTrajectories (v2: aggregate ribbons instead of stroked lines)
# ---------------------------------------------------------------------------

_NEW_DRAW_SAMPLE_TRAJECTORIES = r"""function drawSampleTrajectories(tracingGroup, sampleAssignments, nodes, selectedFlow, data, minFlowSamples) {
        const sampleIds = Object.keys(sampleAssignments);
        console.log('V2: Drawing aligned ribbon trajectories for ' + sampleIds.length + ' samples');

        // Build node lookup
        const nodeMap = new Map();
        nodes.forEach(n => nodeMap.set(n.id, n));

        // Recompute the main flow layout so we can position trajectories WITHIN flow ribbons
        const significantFlows = data.flows.filter(f => f.sampleCount >= minFlowSamples);
        const mainFlowLayout = computeFlowLayout(nodes, significantFlows);

        // Build flow lookup: flowKey -> flow object (for sampleCount)
        const flowLookup = {};
        significantFlows.forEach(flow => {
            const fk = flow.source + '|' + flow.target + '|' + flow.sourceK + '|' + flow.targetK;
            flowLookup[fk] = flow;
        });

        // ---- 1. Aggregate transitions ----
        const transitionAgg = {};
        const segmentCounts  = {};

        sampleIds.forEach(sampleId => {
            const asgn = sampleAssignments[sampleId];
            const sortedKs = Object.keys(asgn).map(k => parseInt(k)).sort((a, b) => a - b);

            sortedKs.forEach(k => {
                const a = asgn[k];
                const segKey = a.topicId + '::' + a.level;
                segmentCounts[segKey] = (segmentCounts[segKey] || 0) + 1;
            });

            for (let ii = 0; ii < sortedKs.length - 1; ii++) {
                const k1 = sortedKs[ii];
                const k2 = sortedKs[ii + 1];
                if (k2 - k1 !== 1) continue;
                const a1 = asgn[k1];
                const a2 = asgn[k2];
                const tKey = a1.topicId + '::' + a1.level + '|' + a2.topicId + '::' + a2.level;
                if (!transitionAgg[tKey]) {
                    transitionAgg[tKey] = {
                        srcTopic: a1.topicId, srcLevel: a1.level,
                        tgtTopic: a2.topicId, tgtLevel: a2.level,
                        srcK: k1, tgtK: k2, count: 0
                    };
                }
                transitionAgg[tKey].count += 1;
            }
        });

        // ---- 2. Draw trajectory ribbons aligned within their flow ribbons ----
        const trajectoryFill   = 'rgba(255, 107, 53, 0.55)';
        const trajectoryStroke = 'rgba(200, 80, 30, 0.70)';

        Object.values(transitionAgg).forEach(tr => {
            const srcNode = nodeMap.get(tr.srcTopic);
            const tgtNode = nodeMap.get(tr.tgtTopic);
            if (!srcNode || !tgtNode) return;

            // Skip the selected flow's own transition (already orange from flow ribbon)
            const isSelectedTransition =
                (tr.srcTopic + '_' + tr.srcLevel) === selectedFlow.source &&
                (tr.tgtTopic + '_' + tr.tgtLevel) === selectedFlow.target &&
                tr.srcK === selectedFlow.sourceK &&
                tr.tgtK === selectedFlow.targetK;
            if (isSelectedTransition) return;

            // Find the matching main flow ribbon
            const flowKey = (tr.srcTopic + '_' + tr.srcLevel) + '|' +
                           (tr.tgtTopic + '_' + tr.tgtLevel) + '|' +
                           tr.srcK + '|' + tr.tgtK;
            const mainLayout = mainFlowLayout.get(flowKey);
            const matchingFlow = flowLookup[flowKey];

            let srcYTop, srcYBot, tgtYTop, tgtYBot;

            if (mainLayout && matchingFlow && matchingFlow.sampleCount > 0) {
                // Position trajectory WITHIN the flow ribbon proportionally
                const ratio = Math.min(1.0, tr.count / matchingFlow.sampleCount);
                const srcH = (mainLayout.sourceYBot - mainLayout.sourceYTop) * ratio;
                const tgtH = (mainLayout.targetYBot - mainLayout.targetYTop) * ratio;
                srcYTop = mainLayout.sourceYTop;
                srcYBot = mainLayout.sourceYTop + srcH;
                tgtYTop = mainLayout.targetYTop;
                tgtYBot = mainLayout.targetYTop + tgtH;
            } else {
                // Fallback: flow too small or not found, use segment bounds
                const srcBounds = getSegmentBounds(srcNode, tr.srcLevel);
                const tgtBounds = getSegmentBounds(tgtNode, tr.tgtLevel);
                const srcSegH = srcBounds.bottom - srcBounds.top;
                const tgtSegH = tgtBounds.bottom - tgtBounds.top;
                srcYTop = srcBounds.top;
                srcYBot = srcBounds.top + Math.max(2, srcSegH * 0.15);
                tgtYTop = tgtBounds.top;
                tgtYBot = tgtBounds.top + Math.max(2, tgtSegH * 0.15);
            }

            const ribbonPath = createRibbonPath(
                srcNode.x + 15, srcYTop, srcYBot,
                tgtNode.x - 15, tgtYTop, tgtYBot
            );

            tracingGroup.append('path')
                .attr('d', ribbonPath)
                .attr('fill', trajectoryFill)
                .attr('stroke', trajectoryStroke)
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.75)
                .style('pointer-events', 'none');
        });

        // ---- 3. Dots at segment centres ----
        const maxCnt = Math.max(...Object.values(segmentCounts), 1);
        Object.entries(segmentCounts).forEach(([segKey, count]) => {
            const parts = segKey.split('::');
            const node = nodeMap.get(parts[0]);
            if (!node) return;
            const y = calculateSegmentY(node, parts[1]);
            const r = 3 + (count / maxCnt) * 5;
            tracingGroup.append('circle')
                .attr('cx', node.x)
                .attr('cy', y)
                .attr('r', r)
                .attr('fill', '#ff6b35')
                .attr('stroke', 'white')
                .attr('stroke-width', 1.5)
                .attr('opacity', 0.8)
                .style('pointer-events', 'none');
        });

        console.log('V2: Aligned ribbon trajectories drawn');
    }

    """

# ---------------------------------------------------------------------------
# Extra CSS for ribbon transitions
# ---------------------------------------------------------------------------

_EXTRA_CSS = """
    .flow-ribbon {
        transition: fill 0.15s ease, opacity 0.15s ease;
    }
"""

# ---------------------------------------------------------------------------
# Build the patched ESM from the parent class
# ---------------------------------------------------------------------------

_original_esm = StripeSankeyInline._esm

# 1. Replace the flow-drawing section
_before_flows, _rest = _original_esm.split("// Calculate flow width scaling", 1)
_old_flow_code, _after_flows = _rest.split("// Create sample tracing layer", 1)
_after_flows = "// Create sample tracing layer" + _after_flows

_patched_esm = _before_flows + _NEW_FLOW_CODE + _after_flows

# 2. Replace drawSampleTrajectories with ribbon-based version
_before_dst, _rest2 = _patched_esm.split("function drawSampleTrajectories(", 1)
_old_dst, _after_dst = _rest2.split("function highlightSampleSegments(", 1)
_after_dst = "function highlightSampleSegments(" + _after_dst

_patched_esm = _before_dst + _NEW_DRAW_SAMPLE_TRAJECTORIES + _after_dst

# 3. Replace updateSampleTracing with dim/restore version
_NEW_UPDATE_SAMPLE_TRACING = r"""function updateSampleTracing(g, data, selectedFlow, nodes, flows, kValues, model) {
        // Clear previous tracing
        g.selectAll(".sample-tracing").selectAll("*").remove();
        g.selectAll(".sample-count-badge").remove();
        g.selectAll(".sample-info-panel").remove();

        // Reset segment highlighting - set all segments back to white borders
        g.selectAll(".nodes rect").attr("stroke", "white").attr("stroke-width", 1);

        if (!selectedFlow || Object.keys(selectedFlow).length === 0) {
            // === RESTORE: un-dim everything ===
            g.selectAll(".flows .flow-ribbon").each(function() {
                const p = d3.select(this);
                const orig = p.attr("data-original-fill");
                if (orig) p.attr("fill", orig);
                p.attr("opacity", 1.0);
            });
            g.selectAll(".nodes rect").each(function() {
                d3.select(this).attr("opacity", 1.0);
            });
            g.selectAll(".nodes text").each(function() {
                d3.select(this).attr("opacity", 1.0);
            });
            return;
        }

        console.log("V2: Tracing samples with dimming for selected flow:", selectedFlow);

        // === DIM: fade non-selected flows and nodes ===
        const selectedFk = selectedFlow.source + '|' + selectedFlow.target + '|' + selectedFlow.sourceK + '|' + selectedFlow.targetK;

        g.selectAll(".flows .flow-ribbon").each(function() {
            const p = d3.select(this);
            if (!p.attr("data-original-fill")) {
                p.attr("data-original-fill", p.attr("fill"));
            }
            if (p.attr("data-flow-key") === selectedFk) {
                // Keep selected flow bright orange
                p.attr("fill", "rgba(255, 107, 53, 0.70)").attr("opacity", 1.0);
            } else {
                // Dim non-selected flows
                p.attr("fill", "rgba(210, 210, 210, 0.15)").attr("opacity", 1.0);
            }
        });

        // Dim all node bars and labels
        g.selectAll(".nodes rect").attr("opacity", 0.25);
        g.selectAll(".nodes text").attr("opacity", 0.25);

        const tracingGroup = g.select(".sample-tracing");
        const samples = selectedFlow.samples || [];
        const sampleIds = samples.map(function(s) { return s.sample; });

        console.log('Tracing ' + sampleIds.length + ' samples');

        if (sampleIds.length === 0) {
            showSampleInfo(g, selectedFlow, 0);
            return;
        }

        // Find where these samples are assigned across all K values
        const sampleAssignments = traceSampleAssignments(sampleIds, data, flows, kValues);

        // Draw sample trajectory paths
        const minFlowSamples = model.get("min_flow_samples") || 10;
        drawSampleTrajectories(tracingGroup, sampleAssignments, nodes, selectedFlow, data, minFlowSamples);

        // Highlight segments containing these samples (restore their opacity)
        highlightSampleSegments(g, sampleAssignments, nodes);

        // Restore opacity on highlighted node segments so they pop out of the dim
        g.selectAll(".nodes rect").each(function() {
            const rect = d3.select(this);
            if (parseInt(rect.attr("stroke-width")) === 3) {
                rect.attr("opacity", 1.0);
            }
        });

        // Show detailed sample info panel
        showSampleInfo(g, selectedFlow, sampleIds.length);
    }

    """

_before_ust, _rest3 = _patched_esm.split("function updateSampleTracing(", 1)
_old_ust, _after_ust = _rest3.split("function drawAllClusterTrajectories(", 1)
_after_ust = "function drawAllClusterTrajectories(" + _after_ust

_patched_esm = _before_ust + _NEW_UPDATE_SAMPLE_TRACING + _after_ust

# 4. Replace drawAllClusterTrajectories with ribbon-based version
_NEW_DRAW_ALL_CLUSTER_TRAJECTORIES = r"""function drawAllClusterTrajectories(g, rawData, nodes, flows, kValues, model) {
        console.log('V2: Drawing ribbon cluster trajectories...');

        const clusterTrajectories = rawData.cluster_trajectories;
        if (!clusterTrajectories || clusterTrajectories.length === 0) {
            console.log('No cluster trajectories found');
            return;
        }

        var clusterGroup = g.select('.cluster-tracing');
        if (clusterGroup.empty()) {
            clusterGroup = g.append('g').attr('class', 'cluster-tracing');
        }
        clusterGroup.selectAll('*').remove();

        var minFlowSamples = model.get('min_flow_samples') || 10;

        // Build node lookup and recompute main flow layout for alignment
        var nodeMap = new Map();
        nodes.forEach(function(n) { nodeMap.set(n.id, n); });

        // Process flow data to match the format computeFlowLayout expects
        var processedFlows = (rawData.flows || []).map(function(f) {
            return {
                source: f.source_segment,
                target: f.target_segment,
                sourceK: f.source_k,
                targetK: f.target_k,
                sampleCount: f.sample_count || 0,
                samples: f.samples || []
            };
        }).filter(function(f) { return f.sampleCount >= minFlowSamples; });

        var mainFlowLayout = computeFlowLayout(nodes, processedFlows);

        // Build flow lookup for sample counts
        var flowLookup = {};
        processedFlows.forEach(function(f) {
            var fk = f.source + '|' + f.target + '|' + f.sourceK + '|' + f.targetK;
            flowLookup[fk] = f;
        });

        console.log('Drawing ' + clusterTrajectories.length + ' clusters as ribbons');

        clusterTrajectories.forEach(function(cluster) {
            var clusterId = cluster.cluster_id;
            var baseColor = cluster.color;
            var sampleList = cluster.samples;
            var nSamples = cluster.n_samples;

            var clusterAlpha = model.get('cluster_alpha') || 0.6;
            var rgbaMatch = baseColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
            var fillColor = rgbaMatch
                ? 'rgba(' + rgbaMatch[1] + ',' + rgbaMatch[2] + ',' + rgbaMatch[3] + ',' + clusterAlpha + ')'
                : baseColor;
            var strokeColor = rgbaMatch
                ? 'rgba(' + rgbaMatch[1] + ',' + rgbaMatch[2] + ',' + rgbaMatch[3] + ',' + Math.min(1, clusterAlpha + 0.15) + ')'
                : baseColor;

            // Trace sample assignments
            var sampleAssignments = traceSampleAssignmentsForCluster(sampleList, rawData, nodes, kValues);

            // Aggregate transitions
            var transitionAgg = {};
            var segmentCounts = {};

            Object.values(sampleAssignments).forEach(function(asgn) {
                var sortedKs = Object.keys(asgn).map(function(k) { return parseInt(k); }).sort(function(a,b) { return a-b; });

                sortedKs.forEach(function(k) {
                    var a = asgn[k];
                    var sk = a.topicId + '::' + a.level;
                    segmentCounts[sk] = (segmentCounts[sk] || 0) + 1;
                });

                for (var ii = 0; ii < sortedKs.length - 1; ii++) {
                    var k1 = sortedKs[ii], k2 = sortedKs[ii+1];
                    if (k2 - k1 !== 1) continue;
                    var a1 = asgn[k1], a2 = asgn[k2];
                    var tKey = a1.topicId + '_' + a1.level + '|' + a2.topicId + '_' + a2.level + '|' + k1 + '|' + k2;
                    if (!transitionAgg[tKey]) {
                        transitionAgg[tKey] = {
                            srcTopic: a1.topicId, srcLevel: a1.level,
                            tgtTopic: a2.topicId, tgtLevel: a2.level,
                            srcK: k1, tgtK: k2, count: 0
                        };
                    }
                    transitionAgg[tKey].count += 1;
                }
            });

            // Draw ribbon trajectories aligned within flow ribbons
            var drawnPaths = 0;
            Object.values(transitionAgg).forEach(function(tr) {
                if (tr.count < minFlowSamples) return;

                var srcNode = nodeMap.get(tr.srcTopic);
                var tgtNode = nodeMap.get(tr.tgtTopic);
                if (!srcNode || !tgtNode) return;

                // Find matching main flow ribbon
                var flowKey = (tr.srcTopic + '_' + tr.srcLevel) + '|' +
                             (tr.tgtTopic + '_' + tr.tgtLevel) + '|' +
                             tr.srcK + '|' + tr.tgtK;
                var mainLayout = mainFlowLayout.get(flowKey);
                var matchingFlow = flowLookup[flowKey];

                var srcYTop, srcYBot, tgtYTop, tgtYBot;

                if (mainLayout && matchingFlow && matchingFlow.sampleCount > 0) {
                    var ratio = Math.min(1.0, tr.count / matchingFlow.sampleCount);
                    var srcH = (mainLayout.sourceYBot - mainLayout.sourceYTop) * ratio;
                    var tgtH = (mainLayout.targetYBot - mainLayout.targetYTop) * ratio;
                    srcYTop = mainLayout.sourceYTop;
                    srcYBot = mainLayout.sourceYTop + srcH;
                    tgtYTop = mainLayout.targetYTop;
                    tgtYBot = mainLayout.targetYTop + tgtH;
                } else {
                    var srcBounds = getSegmentBounds(srcNode, tr.srcLevel);
                    var tgtBounds = getSegmentBounds(tgtNode, tr.tgtLevel);
                    var srcSegH = srcBounds.bottom - srcBounds.top;
                    var tgtSegH = tgtBounds.bottom - tgtBounds.top;
                    var prop = tr.count / nSamples;
                    srcYTop = srcBounds.top;
                    srcYBot = srcBounds.top + Math.max(2, srcSegH * prop);
                    tgtYTop = tgtBounds.top;
                    tgtYBot = tgtBounds.top + Math.max(2, tgtSegH * prop);
                }

                var ribbonPath = createRibbonPath(
                    srcNode.x + 15, srcYTop, srcYBot,
                    tgtNode.x - 15, tgtYTop, tgtYBot
                );

                clusterGroup.append('path')
                    .attr('d', ribbonPath)
                    .attr('fill', fillColor)
                    .attr('stroke', strokeColor)
                    .attr('stroke-width', 0.5)
                    .attr('opacity', 0.75)
                    .attr('class', 'cluster-trajectory cluster-' + clusterId)
                    .style('pointer-events', 'none');

                drawnPaths++;
            });

            console.log('Cluster ' + clusterId + ': drew ' + drawnPaths + ' ribbon paths');

            // Draw dots at segment centres
            var significantSegs = Object.entries(segmentCounts).filter(function(e) { return e[1] >= minFlowSamples; });
            var maxSegCount = Math.max.apply(null, significantSegs.map(function(e) { return e[1]; }).concat([1]));

            significantSegs.forEach(function(entry) {
                var parts = entry[0].split('::');
                var node = nodeMap.get(parts[0]);
                if (!node) return;
                var y = calculateSegmentY(node, parts[1]);
                var r = 2 + (entry[1] / maxSegCount) * 4;

                clusterGroup.append('circle')
                    .attr('cx', node.x)
                    .attr('cy', y)
                    .attr('r', r)
                    .attr('fill', fillColor)
                    .attr('stroke', 'white')
                    .attr('stroke-width', 0.5)
                    .attr('opacity', 0.8)
                    .attr('class', 'cluster-point cluster-' + clusterId)
                    .style('pointer-events', 'none');
            });
        });

        // Add cluster legend
        drawClusterLegend(g, clusterTrajectories, rawData.cluster_metadata);
        console.log('V2: All cluster ribbon trajectories drawn');
    }

    """

_before_dact, _rest4 = _patched_esm.split("function drawAllClusterTrajectories(", 1)
_old_dact, _after_dact = _rest4.split("function traceSampleAssignmentsForCluster(", 1)
_after_dact = "function traceSampleAssignmentsForCluster(" + _after_dact

_patched_esm = _before_dact + _NEW_DRAW_ALL_CLUSTER_TRAJECTORIES + _after_dact

# 5. Inject new helper functions right before the module export
_patched_esm = _patched_esm.replace(
    "    export default { render };",
    _NEW_JS_FUNCTIONS + "\n    export default { render };",
)


# ---------------------------------------------------------------------------
# The V2 widget class
# ---------------------------------------------------------------------------

class StripeSankeyV2(StripeSankeyInline):
    """Improved StripeSankey with proper Sankey-style flow stacking.

    Drop-in replacement for ``StripeSankeyInline`` — same Python API, same
    traitlets, same constructor parameters.

    Key visual improvements
    -----------------------
    * Flows are drawn as **filled ribbons** that stack along the segment edge
      (instead of all converging at the segment centre).
    * Each ribbon's width on the source side is proportional to the source
      segment height, and on the target side to the target segment height.
    * Ribbons are sorted so that flows heading to lower/higher positions
      do not cross unnecessarily.
    """

    _esm = _patched_esm
    _css = StripeSankeyInline._css + _EXTRA_CSS
