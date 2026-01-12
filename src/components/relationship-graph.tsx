'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Home, Play, Pause } from 'lucide-react';
import { Statement, GraphNode, GraphLink, getEntityColor } from '@/lib/types';
import { statementsToGraphData } from '@/lib/statement-parser';

interface RelationshipGraphProps {
  statements: Statement[];
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  type: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  predicate: string;
}

export function RelationshipGraph({ statements }: RelationshipGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [physicsRunning, setPhysicsRunning] = useState(true);
  const [selectedNode, setSelectedNode] = useState<SimNode | null>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || statements.length === 0) return;

    const { nodes: rawNodes, links: rawLinks } = statementsToGraphData(statements);

    if (rawNodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 400;

    // Clear previous content
    svg.selectAll('*').remove();

    // Add defs for arrow markers
    const defs = svg.append('defs');

    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#64748b');

    // Create container group for zoom/pan
    const g = svg.append('g');

    // Prepare simulation data
    const nodes: SimNode[] = rawNodes.map(n => ({
      ...n,
      x: width / 2 + (Math.random() - 0.5) * 200,
      y: height / 2 + (Math.random() - 0.5) * 200,
    }));

    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    const links: SimLink[] = rawLinks
      .filter(l => nodeMap.has(l.source as string) && nodeMap.has(l.target as string))
      .map(l => ({
        source: nodeMap.get(l.source as string)!,
        target: nodeMap.get(l.target as string)!,
        predicate: l.predicate,
      }));

    // Create force simulation
    const simulation = d3.forceSimulation<SimNode>(nodes)
      .force('link', d3.forceLink<SimNode, SimLink>(links)
        .id(d => d.id)
        .distance(120))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40));

    simulationRef.current = simulation;

    // Draw links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', '#94a3b8')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead)');

    // Draw link labels
    const linkLabel = g.append('g')
      .attr('class', 'link-labels')
      .selectAll('text')
      .data(links)
      .enter().append('text')
      .attr('font-size', '10px')
      .attr('fill', '#64748b')
      .attr('text-anchor', 'middle')
      .text(d => d.predicate.length > 20 ? d.predicate.slice(0, 17) + '...' : d.predicate);

    // Draw nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
      .attr('cursor', 'pointer')
      .call(d3.drag<SVGGElement, SimNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }))
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelectedNode(prev => prev?.id === d.id ? null : d);
      });

    // Node circles
    node.append('circle')
      .attr('r', 16)
      .attr('fill', d => getEntityColor(d.type as any))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))');

    // Node labels
    node.append('text')
      .attr('dy', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#1e293b')
      .attr('font-weight', '500')
      .text(d => d.name.length > 15 ? d.name.slice(0, 12) + '...' : d.name);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as SimNode).x!)
        .attr('y1', d => (d.source as SimNode).y!)
        .attr('x2', d => (d.target as SimNode).x!)
        .attr('y2', d => (d.target as SimNode).y!);

      linkLabel
        .attr('x', d => ((d.source as SimNode).x! + (d.target as SimNode).x!) / 2)
        .attr('y', d => ((d.source as SimNode).y! + (d.target as SimNode).y!) / 2 - 5);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Setup zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform.toString());
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    // Click on background to deselect
    svg.on('click', () => setSelectedNode(null));

    return () => {
      simulation.stop();
    };
  }, [statements]);

  // Toggle physics
  useEffect(() => {
    if (simulationRef.current) {
      if (physicsRunning) {
        simulationRef.current.alpha(0.3).restart();
      } else {
        simulationRef.current.stop();
      }
    }
  }, [physicsRunning]);

  const handleZoomIn = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 1.5);
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 0.67);
    }
  };

  const handleResetZoom = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(500).call(zoomRef.current.transform, d3.zoomIdentity);
    }
  };

  if (statements.length === 0) {
    return (
      <div className="graph-container h-[400px] flex items-center justify-center">
        <p className="text-gray-500">Graph will appear after extracting statements</p>
      </div>
    );
  }

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="graph-container h-[400px]"
      >
        <svg ref={svgRef} className="w-full h-full" />

        {/* Selected node info */}
        {selectedNode && (
          <div className="absolute top-3 left-3 bg-white border shadow-md p-3 max-w-xs">
            <div className="flex items-center justify-between mb-2">
              <span
                className="text-xs font-bold uppercase px-2 py-0.5"
                style={{ backgroundColor: getEntityColor(selectedNode.type as any) + '20', color: getEntityColor(selectedNode.type as any) }}
              >
                {selectedNode.type}
              </span>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-gray-400 hover:text-gray-600 text-sm"
              >
                ×
              </button>
            </div>
            <p className="font-semibold text-sm">{selectedNode.name}</p>
          </div>
        )}

        {/* Controls */}
        <div className="absolute bottom-3 right-3 flex flex-col gap-1">
          <button
            onClick={handleZoomIn}
            className="w-8 h-8 bg-white border shadow-sm flex items-center justify-center hover:bg-gray-50"
            title="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleZoomOut}
            className="w-8 h-8 bg-white border shadow-sm flex items-center justify-center hover:bg-gray-50"
            title="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={handleResetZoom}
            className="w-8 h-8 bg-white border shadow-sm flex items-center justify-center hover:bg-gray-50"
            title="Reset view"
          >
            <Home className="w-4 h-4" />
          </button>
          <button
            onClick={() => setPhysicsRunning(!physicsRunning)}
            className="w-8 h-8 bg-white border shadow-sm flex items-center justify-center hover:bg-gray-50"
            title={physicsRunning ? 'Pause physics' : 'Resume physics'}
          >
            {physicsRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 bg-white/90 border p-2 text-xs">
          <div className="font-semibold mb-1">Entity Types</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            {['ORG', 'PERSON', 'GPE', 'EVENT'].map(type => (
              <div key={type} className="flex items-center gap-1">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getEntityColor(type as any) }}
                />
                <span>{type}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
