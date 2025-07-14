import React from "react";
import { HeatmapSeries, XYPlot, XAxis, YAxis, Hint } from "react-vis";

interface HeatmapProps {
  data: { x: number; y: number; color: number }[]; // data points
  width?: number;
  height?: number;
}

const Heatmap: React.FC<HeatmapProps> = ({
  data,
  width = 400,
  height = 300,
}) => {
  const [hoveredNode, setHoveredNode] = React.useState<null | {
    x: number;
    y: number;
    color: number;
  }>(null);

  return (
    <XYPlot
      xType="ordinal"
      yType="ordinal"
      width={width}
      height={height}
      margin={{ left: 50, bottom: 50, right: 10, top: 10 }}
    >
      <XAxis />
      <YAxis />
      <HeatmapSeries
        className="heatmap-series-example"
        colorRange={["#ffffff", "#007bff"]}
        data={data}
        onValueMouseOver={(datapoint) => setHoveredNode(datapoint)}
        onValueMouseOut={() => setHoveredNode(null)}
      />
      {hoveredNode && (
        <Hint value={hoveredNode}>
          <div
            style={{
              background: "black",
              color: "white",
              padding: "5px",
              borderRadius: "3px",
            }}
          >
            {`x: ${hoveredNode.x}, y: ${hoveredNode.y}, value: ${hoveredNode.color}`}
          </div>
        </Hint>
      )}
    </XYPlot>
  );
};

export default Heatmap;

