import React, { useState } from "react";
import axios from "axios";
import * as XLSX from "xlsx";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function ResumeMatcher({ resumeText }) {
  const [jobs, setJobs] = useState([]);
  const [results, setResults] = useState([]);
  const [threshold, setThreshold] = useState(0);

  const addJob = () => setJobs([...jobs, { title: "", description: "" }]);
  const removeJob = (index) => setJobs(jobs.filter((_, i) => i !== index));
  const updateJob = (index, field, value) => {
    const updatedJobs = [...jobs];
    updatedJobs[index][field] = value;
    setJobs(updatedJobs);
  };

  const handleCompare = async () => {
    try {
      const response = await axios.post(
        "https://resume-screener-backend-dk4j.onrender.com/match_multiple",
        {
          resume: resumeText,
          jobs: jobs
        }
      );
      setResults(response.data.results);
    } catch (error) {
      console.error("Error comparing resume:", error);
    }
  };

  const highlightText = (text, keywords) => {
    if (!keywords || keywords.length === 0) return text;
    const regex = new RegExp(`\\b(${keywords.join("|")})\\b`, "gi");
    return text.split(regex).map((part, i) =>
      keywords.includes(part.toLowerCase()) ? (
        <span key={i} style={{ backgroundColor: "yellow", fontWeight: "bold" }}>
          {part}
        </span>
      ) : (
        part
      )
    );
  };

  const exportCSV = () => {
    if (results.length === 0) return;
    const header = ["Job Title", "Match Score (%)", "Matched Keywords"];
    const rows = results.map(r => [
      r.title,
      r.score,
      r.keywords?.join(" | ") || "None"
    ]);
    const csvContent = [header, ...rows].map(e => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "resume_match_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportExcel = () => {
    if (results.length === 0) return;
    const worksheetData = results.map(r => ({
      "Job Title": r.title,
      "Match Score (%)": r.score,
      "Matched Keywords": r.keywords?.join(" | ") || "None"
    }));
    const worksheet = XLSX.utils.json_to_sheet(worksheetData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Results");
    XLSX.writeFile(workbook, "resume_match_results.xlsx");
  };

  const filteredResults = results.filter(r => r.score >= threshold);

  const chartData = {
    labels: filteredResults.map(r => r.title),
    datasets: [
      {
        label: "Match Score (%)",
        data: filteredResults.map(r => r.score),
        backgroundColor: "rgba(75, 192, 192, 0.6)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "Resume Match Scores by Job" }
    }
  };

  return (
    <div>
      <h2>Resume Matcher</h2>
      <button onClick={addJob}>Add Job</button>

      {jobs.map((job, index) => (
        <div key={index} style={{ marginBottom: "15px" }}>
          <input
            type="text"
            placeholder="Job Title"
            value={job.title}
            onChange={(e) => updateJob(index, "title", e.target.value)}
          />
          <br />
          <textarea
            placeholder="Paste Job Description"
            value={job.description}
            onChange={(e) => updateJob(index, "description", e.target.value)}
          />
          <br />
          <button onClick={() => removeJob(index)}>Remove</button>
        </div>
      ))}

      {jobs.length > 0 && (
        <button onClick={handleCompare}>Compare All</button>
      )}

      {results.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <label>
            Minimum Match Score (%):{" "}
            <input
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              style={{ width: "60px", marginBottom: "10px" }}
            />
          </label>

          <table border="1">
            <thead>
              <tr>
                <th>Job Title</th>
                <th>Match Score (%)</th>
                <th>Matched Keywords</th>
              </tr>
            </thead>
            <tbody>
              {filteredResults.map((r, i) => (
                <tr key={i}>
                  <td>{r.title}</td>
                  <td>{r.score}</td>
                  <td>{r.keywords?.length > 0 ? r.keywords.join(", ") : "None"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: "10px" }}>
            <button onClick={exportCSV}>Download CSV</button>
            <button onClick={exportExcel} style={{ marginLeft: "10px" }}>
              Download Excel
            </button>
          </div>

          <div style={{ marginTop: "20px" }}>
            <Bar data={chartData} options={chartOptions} />
          </div>

          <h3>Highlighted Job Descriptions</h3>
          {filteredResults.map((r, i) => (
            <div key={i} style={{ marginBottom: "15px" }}>
              <h4>{r.title}</h4>
              <p>{highlightText(jobs[i].description, r.keywords)}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ResumeMatcher;
