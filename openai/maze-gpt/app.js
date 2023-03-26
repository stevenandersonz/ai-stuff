(function (global) {
  const Maze = function (rows = 4, cols = 4, obstacles = 0) {
    this.rows = rows;
    this.cols = cols;
    this.obstacles = obstacles;
    this.grid = [];
  };
  Maze.prototype = {
    createGrid: function () {
      for (let i = 0; i < this.rows; i++) {
        this.grid.push([]);
        for (let j = 0; j < this.cols; j++) {
          if (Math.random() < this.obstacles) {
            this.grid[i].push("B"); // Create empty spaces inside thegrid
          }
          this.grid[i].push("-"); // Create empty spaces inside thegrid
        }
      }
      this.grid[0][0] = "X";
      this.grid[this.rows - 1][this.cols - 1] = "O";
    },
    // Initialize maze
    renderUI: function () {
      const mazeContainer = document.getElementById("maze-container");
      mazeContainer.innerHTML = "";
      const root = document.documentElement;
      root.style.setProperty("--maze-rows", this.rows);
      root.style.setProperty("--maze-columns", this.cols);
      for (var i = 0; i < this.rows; i++) {
        for (var j = 0; j < this.cols; j++) {
          var cell = document.createElement("div");
          cell.className = "cell";
          cell.id = i + "-" + j;
          document.getElementById("maze-container").appendChild(cell);
        }
      }
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          cell = document.getElementById(`${i}-${j}`);
          if (this.grid[i][j] === "X") {
            cell.classList.add("entry");
          } else if (this.grid[i][j] === "O") {
            cell.classList.add("exit");
          } else if (this.grid[i][j] === "P") {
            cell.classList.add("path");
          } else if (this.grid[i][j] === "B") {
            cell.classList.add("wall");
          }
          mazeContainer.appendChild(cell);
        }
      }
    },
    getGrid: function () {
      return this.grid;
    },
    setRows: function (rows) {
      this.rows = rows;
    },
    setCols: function (cols) {
      this.cols = cols;
    },
    setPath: function (path) {
      for (let [x, y] of path) {
        this.grid[x][y] = "P";
      }
    },
  };
  global.Maze = Maze;
  // Render maze
})(window);

window.addEventListener("load", () => {
  let maze = new Maze();
  maze.createGrid();
  maze.renderUI();
  const findPathBtn = document.getElementById("submit");
  const renderBtn = document.getElementById("render-btn");
  renderBtn.addEventListener("click", () => {
    let rows = Number(document.getElementById("rows-config").value);
    let cols = Number(document.getElementById("cols-config").value);
    let obstaclesProb = Number(
      document.getElementById("obstacle-config").value
    );
    maze = new Maze(rows, cols, obstaclesProb);
    maze.createGrid();
    maze.renderUI();
  });
  findPathBtn.addEventListener("click", async () => {
    const res = await fetch("http://localhost:8080/", {
      method: "POST",
      body: JSON.stringify({ board: maze.getGrid() }),
      headers: {
        "Content-Type": "application/json",
      },
    }).then((res) => res.json());
    maze.setPath(res.path);
    maze.renderUI();
  });
});
