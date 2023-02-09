## Project title

### Description
This project is about awesome stuff.  

It does that with this and that methods.

And here is a simple demo you can run.

### CI
This project is not (yet) configured to have CI running. To configure it, you can copy the `.gitlab-ci.yml` from another repo. Or, uou can use the [CI yml generator](https://tulrfsd.pages.gitlab.lrz.de/common/documentation/fsd/docs/dev-process/gitlab/gitlabci-yml-generator/index.html) or check the [FSD CI documentation](https://tulrfsd.pages.gitlab.lrz.de/common/documentation/fsd/dev-process/ci-setup) to create a basic one.

If your repo is subject to code generation and/or uses `mrails`, you probably want to enforce successful CI pipeline before merge to `dev` or `master`. For that you need to:
- Check `Enable merged results pipelines` and `Enable merge trains` under `Settings/Merge Request`
- Configure CI [rules](https://docs.gitlab.com/ee/ci/yaml/#rules) for the merge trains
  - If you copied the `.gitlab-ci.yml` from another project, chances are the rules are already there
  - You can also check this [example yaml file](https://tulrfsd.pages.gitlab.lrz.de/common/documentation/fsd/dev-process/ci-setup#ci-configuration) where the rules are configured

<br>

----
#### Contact Person
The mainainer <[example@tum.de](mailto:example@tum.de)>
