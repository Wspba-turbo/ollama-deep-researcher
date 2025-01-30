from pyvis.network import Network

# 创建网络图对象
net = Network(height="800px", width="100%", notebook=False, directed=True)
net.toggle_physics(True)
net.show_buttons(filter_=['physics'])

# 添加节点和边
net.add_node(1, label="Node 1", color="#FF6B6B", size=30)
net.add_node(2, label="Node 2", color="#4ECDC4", size=30)
net.add_node(3, label="Node 3", color="#45B7D1", size=30)
net.add_edge(1, 2)
net.add_edge(2, 3)

# 设置物理布局参数
net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -100,
      "springLength": 200,
      "avoidOverlap": 0.5
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based"
  }
}
""")

# 生成HTML文件
net.show("test_pyvis.html", notebook=False)