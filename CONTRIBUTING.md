We happily welcome contributions to this project. We use GitHub Issues to track community reported issues and GitHub Pull Requests for accepting changes pursuant to a CLA.

### Contribution Process:
1. Create a feature branch to work on your contribution
    * If you are a Databricks employee withouth "Contributor" access, create a Github issue tagging @tj.
    * If you are a community contributor, [fork the repo](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).
2. Clone the `template` directory. This contains the mandatory minimum files to contribute a Ray-related code example. 
    * `README.md`: complete the sections of this markdown with solution details
    * `01_notebook_template`: minimum required notebook format. Your contribution can include multiple notebooks.
    * Rename the directory to a descriptive title. The recommended directory naming is: `Solution_Name_Snakecase`, and should include any important "technology" details in the name (e.g. `HPO_ML_Training_Optuna`, `CICD_Deployment_DABS`)
3. Add package license details to main `README.md`
    * If your solutions uses any packages/libraries that are not part of the Databricks Runtime, add a table row with license details at the bottom of the main [README](README.md)
4. Submit a PR with your changes
    * A repo maintainer will be notified and begin review.
    * If changes are needed, they will be indicated on the PR. Once complete, the contribution will be merged into the repo. Congrats!
