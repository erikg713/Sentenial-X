import React from "react";
import { Graph } from "react-d3-graph";

interface Node {
  id: string;
  label?: string;
}

interface Link {
  source: string;
  target: string;
}

interface AttackGraphProps {
  nodes: Node[];
  links: Link[];
  width?: number;
  height?: number;
}

const defaultConfig = {
  nodeHighlightBehavior: true,
  node: {
    color: "red",
    size: 300,
    highlightStrokeColor: "blue",
    labelProperty: "label",
  },
  link: {
    highlightColor: "lightblue",
  },
  directed: true,
  height: 400,
  width: 600,
};

const AttackGraph: React.FC<AttackGraphProps> = ({
  nodes,
  links,
  width = 600,
  height = 400,
}) => {
  const data = {
    nodes: nodes.map((node) => ({ id: node.id, label: node.label || node.id })),
    links,
  };

  const config = { ...defaultConfig, width, height };

  const onClickNode = (nodeId: string) => {
    alert(`Node clicked: ${nodeId}`);
  };

  const onClickLink = (source: string, target: string) => {
    alert(`Link clicked from ${source} to ${target}`);
  };

  return (
    <div>
      <Graph
        id="attack-graph" // id is mandatory, unique per graph
        data={data}
        config={config}
        onClickNode={onClickNode}
        onClickLink={onClickLink}
      />
    </div>
  );
};

export default AttackGraph;

