const cells = document.querySelectorAll(".cell");
const resetButton = document.getElementById("reset");
let currentPlayer = "x";

cells.forEach((cell) => {
  cell.addEventListener("click", async () => {
    if (!cell.classList.contains("x") && !cell.classList.contains("o")) {
      cell.classList.add(currentPlayer);
      cell.textContent = currentPlayer.toUpperCase();
      cell.style.backgroundColor =
        currentPlayer === "x" ? "#f0b3b3" : "#b3c6f0";
      currentPlayer = currentPlayer === "x" ? "o" : "x";
      checkWinner();
      const response = await fetch("http://localhost:8080/", {
        method: "POST",
        body: JSON.stringify({ board: printBoard() }),
        headers: { "Content-Type": "application/json" },
      }).then((res) => res.json());
      let nextMoveCell = document.getElementById(
        `cell-${response.move.row * 3 + response.move.col}`
      );
      // Get a reference to the "trash-talk" element
      const trashTalkEl = document.getElementById("trash-talk");
      // Update the contents of the "trash-talk" element with some text
      const messageEl = document.createElement("p");
      messageEl.textContent = response.trashTalk;
      trashTalkEl.appendChild(messageEl);
      nextMoveCell.classList.add(currentPlayer);
      nextMoveCell.textContent = currentPlayer.toUpperCase();
      nextMoveCell.style.backgroundColor =
        currentPlayer === "x" ? "#f0b3b3" : "#b3c6f0";
      currentPlayer = currentPlayer === "x" ? "o" : "x";
      checkWinner();
    }
  });
});

function checkWinner() {
  const winningCombos = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];

  for (let i = 0; i < winningCombos.length; i++) {
    const [a, b, c] = winningCombos[i];
    const cellA = document.getElementById(`cell-${a}`);
    const cellB = document.getElementById(`cell-${b}`);
    const cellC = document.getElementById(`cell-${c}`);

    if (
      cellA.classList.contains("x") &&
      cellB.classList.contains("x") &&
      cellC.classList.contains("x")
    ) {
      endGame("x");
      return;
    }

    if (
      cellA.classList.contains("o") &&
      cellB.classList.contains("o") &&
      cellC.classList.contains("o")
    ) {
      endGame("o");
      return;
    }
  }

  let isTie = true;

  for (let i = 0; i < cells.length; i++) {
    if (
      !cells[i].classList.contains("x") &&
      !cells[i].classList.contains("o")
    ) {
      isTie = false;
      break;
    }
  }

  if (isTie) {
    endGame("tie");
  }
}

function endGame(winner) {
  const message =
    winner === "tie" ? "It's a tie!" : `Player ${winner.toUpperCase()} wins!`;
  alert(message);
  resetBoard();
}
function printBoard() {
  let board = [];
  cells.forEach((cell) => {
    if (cell.classList.contains("x")) {
      board.push("x");
    } else if (cell.classList.contains("o")) {
      board.push("o");
    } else {
      board.push("-");
    }
  });
  return board;
}
function resetBoard() {
  cells.forEach((cell) => {
    cell.classList.remove("x", "o");
    cell.textContent = "";
    cell.style.backgroundColor = "#fff";
  });
  currentPlayer = "x";
}

resetButton.addEventListener("click", resetBoard);
