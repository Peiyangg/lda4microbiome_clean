"""
stripesankey_v3.py - V2 + proportional metadata sub-ribbon coloring.

Each flow ribbon is split into colored slices proportional to the metadata
composition of its samples.  When no metadata is provided, behaves like V2.

Usage:
    widget = StripeSankeyV3(sankey_data=sankey_data, width=1400, height=700)
    widget.set_metadata_coloring(sankey_data, metadata_df, "Country")
"""

import traitlets
from .stripesankey_v2 import StripeSankeyV2

# Default d3 category10 palette
_DEFAULT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

# ---------------------------------------------------------------------------
# V3 flow drawing code (replaces V2's flow section)
# ---------------------------------------------------------------------------

_V3_FLOW_CODE = r"""// === V3: Stacked flow layout with proportional metadata sub-ribbons ===
        const flowLayout = computeFlowLayout(nodes, significantFlows);
        const _metaBreakdown = model.get("metadata_breakdown") || {};
        const _metaLegend    = model.get("metadata_legend") || {};

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

                const isSelected = selectedFlow &&
                    selectedFlow.source === flow.source &&
                    selectedFlow.target === flow.target &&
                    selectedFlow.sourceK === flow.sourceK &&
                    selectedFlow.targetK === flow.targetK;

                const selectedFill   = "rgba(255, 107, 53, 0.65)";
                const selectedStroke = "rgba(200, 80, 30, 0.80)";
                const defaultFill    = "rgba(180, 180, 180, 0.40)";
                const defaultStroke  = "rgba(150, 150, 150, 0.55)";

                const breakdown = _metaBreakdown[fk];
                const srcH = layout.sourceYBot - layout.sourceYTop;
                const tgtH = layout.targetYBot - layout.targetYTop;

                if (breakdown && breakdown.length > 0 && !isSelected) {
                    // --- Draw outer border for the whole flow first ---
                    var outerPath = createRibbonPath(
                        sourceNode.x + 15, layout.sourceYTop, layout.sourceYBot,
                        targetNode.x - 15, layout.targetYTop, layout.targetYBot
                    );
                    flowGroup.append("path")
                        .attr("d", outerPath)
                        .attr("data-flow-key", fk)
                        .attr("fill", "none")
                        .attr("stroke", "rgba(100,100,100,0.35)")
                        .attr("stroke-width", 1.2)
                        .attr("class", "flow-" + flowIndex + " flow-ribbon flow-border")
                        .style("pointer-events", "none");

                    // --- Draw proportional sub-ribbons ---
                    var srcOff = 0, tgtOff = 0;
                    breakdown.forEach(function(entry) {
                        var prop = entry.proportion;
                        var subSrcH = srcH * prop;
                        var subTgtH = tgtH * prop;
                        var subPath = createRibbonPath(
                            sourceNode.x + 15,
                            layout.sourceYTop + srcOff,
                            layout.sourceYTop + srcOff + subSrcH,
                            targetNode.x - 15,
                            layout.targetYTop + tgtOff,
                            layout.targetYTop + tgtOff + subTgtH
                        );
                        var c = d3.color(entry.color);
                        c.opacity = 0.55;
                        var fillStr = c.toString();

                        flowGroup.append("path")
                            .attr("d", subPath)
                            .attr("data-flow-key", fk)
                            .attr("fill", fillStr)
                            .attr("stroke", "rgba(255,255,255,0.25)")
                            .attr("stroke-width", 0.3)
                            .attr("opacity", 1.0)
                            .attr("class", "flow-" + flowIndex + " flow-ribbon")
                            .style("cursor", "pointer")
                            .on("mouseover", function(event) {
                                if (!isSelected) {
                                    flowGroup.selectAll('[data-flow-key="' + fk + '"]')
                                        .attr("stroke", "#333").attr("stroke-width", 1);
                                }
                                showTooltip(g, event, flow);
                            })
                            .on("mouseout", function() {
                                if (!isSelected) {
                                    flowGroup.selectAll('[data-flow-key="' + fk + '"]')
                                        .attr("stroke", "rgba(255,255,255,0.25)").attr("stroke-width", 0.3);
                                }
                                g.selectAll(".tooltip").remove();
                            })
                            .on("click", function(event) {
                                event.stopPropagation();
                                if (isSelected) {
                                    model.set("selected_flow", {});
                                } else {
                                    model.set("selected_flow", {
                                        source: flow.source, target: flow.target,
                                        sourceK: flow.sourceK, targetK: flow.targetK,
                                        samples: flow.samples, sampleCount: flow.sampleCount
                                    });
                                }
                                model.save_changes();
                            });

                        srcOff += subSrcH;
                        tgtOff += subTgtH;
                    });
                } else {
                    // --- Single ribbon (no metadata or selected) ---
                    var ribbonPath = createRibbonPath(
                        sourceNode.x + 15, layout.sourceYTop, layout.sourceYBot,
                        targetNode.x - 15, layout.targetYTop, layout.targetYBot
                    );
                    var fill   = isSelected ? selectedFill   : defaultFill;
                    var stroke = isSelected ? selectedStroke : defaultStroke;

                    flowGroup.append("path")
                        .attr("d", ribbonPath)
                        .attr("data-flow-key", fk)
                        .attr("fill", fill)
                        .attr("stroke", stroke)
                        .attr("stroke-width", 0.5)
                        .attr("opacity", isSelected ? 0.9 : 1.0)
                        .attr("class", "flow-" + flowIndex + " flow-ribbon")
                        .style("cursor", "pointer")
                        .on("mouseover", function(event) {
                            if (!isSelected) {
                                d3.select(this).attr("fill", "rgba(160,160,160,0.55)");
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
                            if (isSelected) {
                                model.set("selected_flow", {});
                            } else {
                                model.set("selected_flow", {
                                    source: flow.source, target: flow.target,
                                    sourceK: flow.sourceK, targetK: flow.targetK,
                                    samples: flow.samples, sampleCount: flow.sampleCount
                                });
                            }
                            model.save_changes();
                        });
                }
            }
        });

        // Draw metadata legend if available
        if (Object.keys(_metaLegend).length > 0) {
            drawMetadataLegend(g, _metaLegend, width, height);
        }

        """

# ---------------------------------------------------------------------------
# Metadata legend JS function
# ---------------------------------------------------------------------------

_METADATA_LEGEND_FUNCTION = r"""
    // === V3: Metadata category legend ===
    function drawMetadataLegend(g, legendData, chartWidth, chartHeight) {
        g.selectAll(".metadata-legend").remove();

        var entries = Object.entries(legendData);
        if (entries.length === 0) return;

        var legendX = chartWidth + 80;
        var legendY = 0;

        var legend = g.append("g")
            .attr("class", "metadata-legend")
            .attr("transform", "translate(" + legendX + "," + legendY + ")");

        legend.append("text")
            .attr("x", 0).attr("y", 0)
            .style("font-size", "11px")
            .style("font-weight", "bold")
            .style("fill", "#333")
            .text("Metadata");

        entries.forEach(function(entry, i) {
            var cat   = entry[0];
            var color = entry[1];
            var y = 14 + i * 18;

            legend.append("rect")
                .attr("x", 0).attr("y", y - 9)
                .attr("width", 12).attr("height", 12)
                .attr("fill", color)
                .attr("stroke", "#333")
                .attr("stroke-width", 0.5)
                .attr("rx", 1);

            legend.append("text")
                .attr("x", 18).attr("y", y)
                .style("font-size", "10px")
                .style("fill", "#333")
                .text(cat);
        });
    }
"""

# ---------------------------------------------------------------------------
# Build the V3 ESM from V2
# ---------------------------------------------------------------------------

_v2_esm = StripeSankeyV2._esm

# 1. Replace the flow-drawing section with V3's proportional sub-ribbons
_before, _rest = _v2_esm.split("// === V2: Stacked flow layout", 1)
_old, _after = _rest.split("// Create sample tracing layer", 1)
_after = "// Create sample tracing layer" + _after

_v3_esm = _before + _V3_FLOW_CODE + _after

# 2. Inject metadata legend function before export
_v3_esm = _v3_esm.replace(
    "    export default { render };",
    _METADATA_LEGEND_FUNCTION + "\n    export default { render };",
)


# ---------------------------------------------------------------------------
# The V3 widget class
# ---------------------------------------------------------------------------

class StripeSankeyV3(StripeSankeyV2):
    """V2 + proportional metadata sub-ribbon coloring.

    Each flow ribbon is split into colored slices showing the metadata
    composition.  When ``metadata_breakdown`` is empty, behaves like V2.
    """

    _esm = _v3_esm

    # Traitlets synced to JS
    metadata_breakdown = traitlets.Dict(default_value={}).tag(sync=True)
    metadata_legend = traitlets.Dict(default_value={}).tag(sync=True)

    def set_metadata_coloring(self, sankey_data, metadata_df, column,
                              color_map=None):
        """Color each flow by proportional metadata composition.

        Parameters
        ----------
        sankey_data : dict
            The sankey data dict.
        metadata_df : pandas.DataFrame
            Metadata table (sample IDs as index).
        column : str
            Metadata column to color by.
        color_map : dict, optional
            Category -> hex color.  Auto-generated if None.

        Returns
        -------
        dict
            category -> hex color mapping.
        """
        categories = sorted(metadata_df[column].dropna().unique())
        if color_map is None:
            color_map = {
                cat: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
                for i, cat in enumerate(categories)
            }

        sample_to_cat = metadata_df[column].to_dict()

        breakdown = {}
        for flow in sankey_data.get("flows", []):
            samples = flow.get("samples", [])
            if not samples:
                continue

            cat_counts = {}
            for s in samples:
                cat = sample_to_cat.get(s.get("sample", ""), None)
                if cat is not None:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

            if not cat_counts:
                continue

            total = sum(cat_counts.values())
            # Sort by count descending for consistent stacking
            entries = sorted(cat_counts.items(), key=lambda x: -x[1])
            flow_breakdown = [
                {
                    "category": cat,
                    "proportion": round(count / total, 4),
                    "color": color_map.get(cat, "#888888"),
                }
                for cat, count in entries
            ]

            fk = (
                f"{flow['source_segment']}|{flow['target_segment']}"
                f"|{flow['source_k']}|{flow['target_k']}"
            )
            breakdown[fk] = flow_breakdown

        self.metadata_breakdown = breakdown
        self.metadata_legend = color_map
        print(
            f"Metadata coloring applied: {len(breakdown)} flows, "
            f"'{column}' ({len(categories)} categories)"
        )
        return color_map

    def clear_metadata_coloring(self):
        """Remove metadata coloring, revert to grey."""
        self.metadata_breakdown = {}
        self.metadata_legend = {}
